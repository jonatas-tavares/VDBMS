import os
import glob
# import pandas as pd # Pandas não será mais usado para leitura de TSV
import polars as pl # Importar Polars
from pymilvus import connections, utility, Collection, CollectionSchema, FieldSchema, DataType
from sentence_transformers import SentenceTransformer
import logging

# --- Configuration ---
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
MILVUS_ALIAS = "default" # Connection alias

COLLECTION_NAME = "document_embeddings_collection"

# Embedding model:
# 'all-MiniLM-L6-v2' is a good general-purpose English model (384 dimensions)
# 'paraphrase-multilingual-MiniLM-L12-v2' is good for multiple languages (384 dimensions)
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
EMBEDDING_DIM = 384  # Must match the chosen model's output dimension

TSV_FILES_DIRECTORY = "./tsv_data/"  # Directory containing your TSV files
TEXT_COLUMN_IN_TSV = "text_content"  # Name of the column in your TSV files that contains the text to embed

# Milvus field names
ID_FIELD_NAME = "id"
TEXT_FIELD_NAME = "original_text"
VECTOR_FIELD_NAME = "embedding"

MAX_TEXT_LENGTH = 4000  # Maximum length for the VARCHAR field storing original text. Adjust if needed.
                               # Milvus has a limit (e.g., 65535 bytes), but shorter is often more practical.

PROCESSING_BATCH_SIZE = 128  # How many rows from TSV to process at a time for embedding
MILVUS_INSERT_BATCH_SIZE = 1000 # How many records to insert into Milvus in one go (Milvus SDK might handle its own batching too)

# Index parameters for the vector field
INDEX_FIELD_NAME = VECTOR_FIELD_NAME
INDEX_METRIC_TYPE = "L2"  # Or "IP" (Inner Product), depending on your embedding model and use case
INDEX_TYPE = "IVF_FLAT"   # A common index type. Alternatives: HNSW, FLAT
INDEX_PARAMS = {"nlist": 128} # Example params for IVF_FLAT. Adjust based on data size and performance needs.

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def connect_to_milvus():
    """Establishes a connection to the Milvus server."""
    try:
        logging.info(f"Connecting to Milvus at {MILVUS_HOST}:{MILVUS_PORT}...")
        connections.connect(alias=MILVUS_ALIAS, host=MILVUS_HOST, port=MILVUS_PORT)
        logging.info("Successfully connected to Milvus.")
    except Exception as e:
        logging.error(f"Failed to connect to Milvus: {e}")
        raise

def create_milvus_collection_if_not_exists():
    """Creates the Milvus collection if it doesn't already exist."""
    if utility.has_collection(COLLECTION_NAME, using=MILVUS_ALIAS):
        logging.info(f"Collection '{COLLECTION_NAME}' already exists.")
        return Collection(COLLECTION_NAME, using=MILVUS_ALIAS)

    logging.info(f"Collection '{COLLECTION_NAME}' does not exist. Creating...")
    
    # Define fields
    field_id = FieldSchema(name=ID_FIELD_NAME, dtype=DataType.INT64, is_primary=True, auto_id=True)
    field_text = FieldSchema(name=TEXT_FIELD_NAME, dtype=DataType.VARCHAR, max_length=MAX_TEXT_LENGTH)
    field_embedding = FieldSchema(name=VECTOR_FIELD_NAME, dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
    
    schema = CollectionSchema(
        fields=[field_id, field_text, field_embedding],
        description="Collection to store text documents and their embeddings",
        enable_dynamic_field=False # Set to True if you want to add fields on the fly without schema changes
    )
    
    collection = Collection(COLLECTION_NAME, schema=schema, using=MILVUS_ALIAS)
    logging.info(f"Collection '{COLLECTION_NAME}' created successfully.")
    
    # Create an index for the vector field for efficient searching
    logging.info(f"Creating index for field '{INDEX_FIELD_NAME}'...")
    index = {
        "index_type": INDEX_TYPE,
        "metric_type": INDEX_METRIC_TYPE,
        "params": INDEX_PARAMS,
    }
    collection.create_index(INDEX_FIELD_NAME, index)
    logging.info(f"Index created successfully on field '{INDEX_FIELD_NAME}'.")
    
    # Load the collection into memory for searching (optional, but good for performance)
    # collection.load() # Loading is usually done before searching, not necessarily right after creation/insertion.
    # For this script, we are primarily inserting. Loading will be needed for search operations later.
    return collection

def load_embedding_model():
    """Loads the sentence embedding model."""
    try:
        logging.info(f"Loading sentence transformer model: '{EMBEDDING_MODEL_NAME}'...")
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        # Check if the model's embedding dimension matches the configuration
        if model.get_sentence_embedding_dimension() != EMBEDDING_DIM:
            logging.warning(
                f"Model '{EMBEDDING_MODEL_NAME}' has dimension "
                f"{model.get_sentence_embedding_dimension()}, but EMBEDDING_DIM is set to {EMBEDDING_DIM}. "
                "Please ensure these match."
            )
            # You might want to raise an error here or dynamically set EMBEDDING_DIM
        logging.info("Embedding model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Failed to load embedding model: {e}")
        raise

def process_tsv_and_insert(filepath, collection, model):
    """Reads a TSV file, generates embeddings, and inserts data into Milvus."""
    logging.info(f"Processing TSV file: {filepath}")
    try:
        data_to_insert = []
        processed_rows_in_file = 0
        # Modificado para usar Polars com iter_batches
        # Nota: Polars' read_csv por padrão infere o separador, mas é bom ser explícito com separator='\t'
        # O batch_size no Polars é o número de linhas por batch.
        for chunk_df in pl.read_csv(filepath, separator='\t', batch_size=PROCESSING_BATCH_SIZE, infer_schema_length=1000, has_header=True, try_parse_dates=False, ignore_errors=True):
            if TEXT_COLUMN_IN_TSV not in chunk_df.columns:
                logging.warning(f"Column '{TEXT_COLUMN_IN_TSV}' not found in {filepath}. Skipping this file/chunk.")
                continue

            # No Polars, para obter uma lista de strings de uma coluna:
            texts = chunk_df[TEXT_COLUMN_IN_TSV].to_list()
            if not texts:
                logging.info(f"No text data found in current batch of {filepath}. Skipping.")
                continue

            logging.info(f"Generating embeddings for {len(texts)} texts from {filepath}...")
            embeddings = model.encode(texts, show_progress_bar=False) # Set show_progress_bar=True for visual feedback
            
            # Prepare data for Milvus insertion
            # Schema: id (auto), original_text, embedding
            # Data format: [[text1, embedding1], [text2, embedding2], ...]
            for text, embedding in zip(texts, embeddings):
                # Truncate text if it's longer than MAX_TEXT_LENGTH to avoid Milvus errors
                truncated_text = text[:MAX_TEXT_LENGTH]
                if len(text) > MAX_TEXT_LENGTH:
                    logging.warning(f"Text truncated for insertion: '{text[:50]}...'")
                
                data_to_insert.append([truncated_text, embedding.tolist()])
            
            processed_rows_in_file += len(texts)

            # Insert in batches to Milvus
            if len(data_to_insert) >= MILVUS_INSERT_BATCH_SIZE:
                logging.info(f"Inserting batch of {len(data_to_insert)} records into Milvus...")
                collection.insert(data_to_insert)
                data_to_insert = [] # Clear batch

        # Insert any remaining data
        if data_to_insert:
            logging.info(f"Inserting remaining {len(data_to_insert)} records into Milvus...")
            collection.insert(data_to_insert)
        
        logging.info(f"Finished processing {processed_rows_in_file} rows from {filepath}.")
        
        # It's good practice to flush after processing a file or a significant number of inserts
        # to ensure data is written to disk segments.
        logging.info(f"Flushing collection '{COLLECTION_NAME}' to persist data from {filepath}...")
        collection.flush()
        logging.info(f"Collection '{COLLECTION_NAME}' flushed.")

    except FileNotFoundError:
        logging.error(f"TSV file not found: {filepath}")
    # Polars pode levantar exceções diferentes para arquivos vazios ou malformados.
    # pl.exceptions.NoDataError é uma possibilidade para arquivos vazios.
    except Exception as e:
        logging.error(f"Error processing file {filepath}: {e}")

def main():
    """Main function to orchestrate the process."""
    logging.info("Starting TSV to Milvus embedding script.")
    
    connect_to_milvus()
    embedding_model = load_embedding_model()
    milvus_collection = create_milvus_collection_if_not_exists()
    
    # Find all .tsv files in the specified directory
    tsv_files = glob.glob(os.path.join(TSV_FILES_DIRECTORY, "*.tsv"))
    
    if not tsv_files:
        logging.warning(f"No .tsv files found in directory: {TSV_FILES_DIRECTORY}")
        return
        
    logging.info(f"Found {len(tsv_files)} TSV files to process: {tsv_files}")
    
    for tsv_file_path in tsv_files:
        process_tsv_and_insert(tsv_file_path, milvus_collection, embedding_model)
        
    # Optional: Get collection entity count after processing all files
    # Milvus counts can be eventually consistent, so this might not be exact immediately after flush.
    # For an exact count after all operations, you might need to wait or use specific Milvus calls.
    # milvus_collection.load() # Ensure collection is loaded for num_entities
    # logging.info(f"Total entities in collection '{COLLECTION_NAME}': {milvus_collection.num_entities}")
    
    logging.info("Script finished processing all TSV files.")

if __name__ == "__main__":
    # Create the tsv_data directory if it doesn't exist, for example purposes
    if not os.path.exists(TSV_FILES_DIRECTORY):
        os.makedirs(TSV_FILES_DIRECTORY)
        logging.info(f"Created directory {TSV_FILES_DIRECTORY}. Please add your TSV files there.")
        # Example: Create a dummy TSV file for testing
        dummy_tsv_content = f"{TEXT_COLUMN_IN_TSV}\tother_column\nHello Milvus!\tdata1\nThis is a test.\tdata2"
        with open(os.path.join(TSV_FILES_DIRECTORY, "example.tsv"), "w") as f:
            f.write(dummy_tsv_content)
        logging.info(f"Created dummy file {os.path.join(TSV_FILES_DIRECTORY, 'example.tsv')} for testing.")

    main()