import numpy as np
from pathlib import Path
import logging
import os
import pickle
from typing import Tuple, List, Any, Union
from sklearn.preprocessing import StandardScaler

class AllosericSitePredictor:
    def __init__(self, model_path: str, labels_path: str, features_path: str):
        self.logger = self._setup_logger()
        self.weights = self._load_pickle(model_path, "model weights")
        self.labels = self._load_pickle(labels_path, "labels")
        self.features = self._load_pickle(features_path, "features")
        self.scaler = StandardScaler()
        
        self._process_weights()
        
        if isinstance(self.features, np.ndarray):
            self.scaler.fit(self.features)
            self.logger.info(f"Scaler fitted with training features of shape {self.features.shape}")

    def _process_weights(self):
        if not isinstance(self.weights, np.ndarray):
            self.weights = np.array(self.weights)
        self.logger.info(f"Original weights shape: {self.weights.shape}")
        self.weights = np.mean(self.weights, axis=0)
        self.logger.info(f"Computed mean weight vector of shape: {self.weights.shape}")

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('AllosericSitePredictor')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def _load_pickle(self, file_path: str, file_type: str) -> Any:
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"{file_type} file not found at {file_path}")
            
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                self.logger.info(f"Loaded {file_type} with type {type(data)}")
                if isinstance(data, np.ndarray):
                    self.logger.info(f"{file_type} shape: {data.shape}")
                elif isinstance(data, list):
                    self.logger.info(f"{file_type} length: {len(data)}")
                return data
        except Exception as e:
            self.logger.error(f"Error loading {file_type} file: {str(e)}")
            raise

    def run_fpocket(self, pdb_file: str) -> str:
        pdb_path = Path(pdb_file)
        if not pdb_path.exists():
            raise FileNotFoundError(f"PDB file not found at {pdb_path}")
        
        base_name = pdb_path.stem
        output_dir = f"{base_name}_out"
        
        self.logger.info(f"Running fpocket on {pdb_file}")
        exit_code = os.system(f"fpocket -f {pdb_file}")
        if exit_code != 0:
            raise RuntimeError(f"fpocket failed with exit code {exit_code}")
            
        self.logger.info("fpocket completed successfully")
        return output_dir

    def _convert_to_numpy(self, features: Union[List, np.ndarray]) -> np.ndarray:
        if isinstance(features, list):
            self.logger.info(f"Converting list of {len(features)} features to numpy array")
            features = np.array(features)
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
            self.logger.info(f"Reshaped features to {features.shape}")
        return features

    def extract_pocket_features(self, info_file: str) -> np.ndarray:
        try:
            from utils.pocket_feature import pocket_feature
            features = pocket_feature(info_file)
            features = self._convert_to_numpy(features)
            self.logger.info(f"Extracted features with shape: {features.shape}")
            scaled_features = self.scaler.transform(features)
            self.logger.info(f"Scaled features with shape: {scaled_features.shape}")
            return scaled_features
        except Exception as e:
            self.logger.error(f"Error extracting pocket features: {str(e)}")
            raise

    def compute_scores(self, features: np.ndarray) -> np.ndarray:
        """Compute scores for each pocket using the mean weight vector."""
        self.logger.info(f"Computing scores for features of shape {features.shape}")
        self.logger.info(f"Using weight vector of shape {self.weights.shape}")
        
        # Compute scores using dot product with mean weight vector
        scores = features @ self.weights
        
        # Convert to probabilities using softmax
        exp_scores = np.exp(scores - np.max(scores))
        probabilities = exp_scores / exp_scores.sum()
        
        # Get top 3 pockets and their probabilities
        top_indices = np.argsort(probabilities)[::-1][:3]
        top_probabilities = probabilities[top_indices]
        
        print("\nTop 3 Predicted Allosteric Sites:")
        print("---------------------------------")
        for i, (idx, prob) in enumerate(zip(top_indices, top_probabilities)):
            print(f"Rank {i+1}: Pocket {idx+1:2d} - Probability: {prob:.4f}")
        print("---------------------------------")
        
        return probabilities

    def predict(self, pdb_file: str) -> Tuple[int, float, List[Tuple[int, float]]]:
        try:
            output_dir = self.run_fpocket(pdb_file)
            info_file = Path(output_dir) / f"{Path(pdb_file).stem}_info.txt"
            
            pocket_features = self.extract_pocket_features(str(info_file))
            probabilities = self.compute_scores(pocket_features)
            
            # Get top 3 pockets and probabilities
            top_indices = np.argsort(probabilities)[::-1][:3]
            top_probabilities = probabilities[top_indices]
            top_predictions = [(idx + 1, prob) for idx, prob in zip(top_indices, top_probabilities)]
            
            return top_indices[0] + 1, top_probabilities[0], top_predictions
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            raise

def main():
    try:
        predictor = AllosericSitePredictor(
            model_path="2023new_features.pkl",
            labels_path="../data/classification/labels.pkl",
            features_path="/home/c00jsw00/allosteric_inhibitor/PASSerRank/src/2023new_features.pkl"
        )
        
        pdb_file = "/home/c00jsw00/allosteric_inhibitor/PASSerRank/src/recn.pdb"
        pocket_index, probability, top_predictions = predictor.predict(pdb_file)
        
        print("\nPrediction Summary:")
        print("---------------------------------")
        print(f"Best prediction: Pocket {pocket_index}")
        print(f"Confidence score: {probability:.4f}")
        print("---------------------------------")
        
    except Exception as e:
        logging.error(f"Program failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
