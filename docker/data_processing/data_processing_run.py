import sys
import logging
import argparse
from data_processing import Data_Processing

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description='Data Processing in Kubernetes')
    parser.add_argument('--config-path', default='/app/dags/config.json', 
                        help='Path to config file')
    parser.add_argument('--data-context-path', default='/app/dags/data/data_context.json', 
                        help='Path to data context file')
    
    args = parser.parse_args()
    
    try:
        logging.info("Starting data processing task...")
        logging.info(f"Config path: {args.config_path}")
        logging.info(f"Data context path: {args.data_context_path}")
        
        # Initialize data processing object
        data_processor = Data_Processing(
            config_path=args.config_path,
            data_context_path=args.data_context_path
        )
        
        # Process data
        result = data_processor.data_processing()
        
        return result
        
    except Exception as e:
        logging.error(f"Error in data processing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()