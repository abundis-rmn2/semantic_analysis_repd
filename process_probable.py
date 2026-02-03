import argparse
import pandas as pd
import os
import logging
from probable_processor import ProbableProcessor
from graph_utils import build_affinity_graph, save_graph

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Probable: Exploratory Disappearance Pipeline")
    parser.add_argument("--input", type=str, default="data/sisovid.csv", help="Input CSV file")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    parser.add_argument("--sample", type=int, default=None, help="Number of records to sample")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM calls")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for processing (not fully implemented in loops yet)")
    
    args = parser.parse_args()

    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        return

    # Load data
    df = pd.read_csv(args.input)
    if args.sample:
        df = df.sample(min(args.sample, len(df))).copy()
        logger.info(f"Sampled {len(df)} records.")

    # Convert dates
    if 'fecha_desaparicion' in df.columns:
        df['date_dt'] = pd.to_datetime(df['fecha_desaparicion'], errors='coerce')

    # Initialize Processor
    processor = ProbableProcessor(output_dir=args.output_dir)
    
    # Run Pipeline
    results = processor.run_pipeline(df, skip_llm=args.skip_llm, batch_size=args.batch_size)
    
    logger.info(f"Pipeline finished. See {args.output_dir}/ for results.")

if __name__ == "__main__":
    main()
