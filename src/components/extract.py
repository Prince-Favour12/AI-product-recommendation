import pandas as pd
from loguru import logger
from dataclasses import dataclass, field
from ..config.setting import settings
import json
from pathlib import Path

@dataclass
class ExtractData:
    path: str = field(default=settings.DATA_PATH)

    def _store_info(self, data: pd.DataFrame) -> None:
        """Store metadata information about the extracted data.

        Args:
            data (pd.DataFrame): The extracted data.
        """
        info = {
            "num_rows": data.shape[0],
            "num_columns": data.shape[1],
            "columns": data.columns.tolist()
        }
        info_path = Path(settings.METADATA_PATH) / "data_info.json"
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=4)
        logger.info(f"Data metadata stored at {info_path}")

    def extract(self) -> pd.DataFrame:
        """Extract data from a CSV file.

        Returns:
            pd.DataFrame: Extracted data as a pandas DataFrame.
        """
        try:
            data = pd.read_csv(self.path)
            logger.info(f"Data extracted successfully from {self.path}")
            self._store_info(data)
            return data
        except Exception as e:
            logger.error(f"Error extracting data from {self.path}: {e}")
            raise