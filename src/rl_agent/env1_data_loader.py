# data_loader.py
"""
Contract Data Loader Module for RL Environment
Handles loading and managing smart contract path databases
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import logging
from dataclasses import dataclass
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ContractData:
    """ساختار داده برای نگهداری اطلاعات یک contract"""
    address: str
    label: str  # safe/vulnerable
    paths: List[Dict]
    profile: Dict
    total_paths: int

    def __post_init__(self):
        """Validation after initialization"""
        assert self.address, "Contract address cannot be empty"
        assert self.label in ['safe', 'vulnerable'], f"Invalid label: {self.label}"
        assert len(self.paths) > 0, "Contract must have at least one path"
        logger.debug(f"ContractData initialized: {self.address} with {len(self.paths)} paths")


class ContractDataLoader:
    """
    مسئول خواندن و مدیریت داده‌های contracts
    این کلاس بخشی از محیط RL است و داده‌های مورد نیاز را فراهم می‌کند
    """

    def __init__(self,
                 # path_db_dir: str = r"C:\Users\Hadis\Documents\NewModel1\path_databases_updated",
                 # profile_dir: str = r"C:\Users\Hadis\Documents\NewModel1\contract_profiles",

                 path_db_dir: str = r"C:\Users\Hadis\Documents\NewModel1\path_databases_updated",
                 profile_dir: str = r"C:\Users\Hadis\Documents\NewModel1\contract_profiles",
                 min_paths_required: int = 1):




        """
        Args:
            path_db_dir: مسیر فولدر path databases (با modifier info)
            profile_dir: مسیر فولدر contract profiles
            min_paths_required: حداقل paths مورد نیاز برای valid بودن contract
        """
        self.path_db_dir = Path(path_db_dir)
        self.profile_dir = Path(profile_dir)
        self.min_paths_required = min_paths_required

        # Validate directories
        self._validate_directories()

        # Cache for performance
        self.loaded_contracts: Dict[str, ContractData] = {}

        # List of valid contracts (with paths)
        self.valid_contracts: List[Tuple[str, int, str]] = []
        self._initialize_valid_contracts()

        logger.info(f"DataLoader initialized with {len(self.valid_contracts)} valid contracts")

    def _validate_directories(self):
        """بررسی وجود directories"""
        if not self.path_db_dir.exists():
            raise FileNotFoundError(f"Path database directory not found: {self.path_db_dir}")
        if not self.profile_dir.exists():
            raise FileNotFoundError(f"Profile directory not found: {self.profile_dir}")

        path_files = list(self.path_db_dir.glob("*_path_database.json"))
        profile_files = list(self.profile_dir.glob("*_profile.json"))

        if len(path_files) == 0:
            raise ValueError(f"No path database files found in {self.path_db_dir}")
        if len(profile_files) == 0:
            raise ValueError(f"No profile files found in {self.profile_dir}")

        logger.debug(f"Found {len(path_files)} path files and {len(profile_files)} profile files")

    def _initialize_valid_contracts(self):
        """شناسایی و ذخیره contracts معتبر"""
        all_addresses = self._get_all_contract_addresses()

        for addr in all_addresses:
            try:
                # Quick check without full loading
                path_file = self.path_db_dir / f"{addr}_path_database.json"
                profile_file = self.profile_dir / f"{addr}_profile.json"

                if path_file.exists() and profile_file.exists():
                    # Load minimal info to check validity
                    with open(path_file, 'r', encoding='utf-8') as f:
                        path_data = json.load(f)
                    with open(profile_file, 'r', encoding='utf-8') as f:
                        profile_data = json.load(f)

                    paths = path_data.get('paths', [])
                    if len(paths) >= self.min_paths_required:
                        label = profile_data.get('label', 'unknown')
                        if label in ['safe', 'vulnerable']:
                            self.valid_contracts.append((addr, len(paths), label))

            except Exception as e:
                logger.warning(f"Error checking contract {addr}: {e}")
                continue

        # Sort by path count (descending)
        self.valid_contracts.sort(key=lambda x: x[1], reverse=True)
        logger.info(f"Found {len(self.valid_contracts)} valid contracts")

    def _get_all_contract_addresses(self) -> List[str]:
        """لیست همه contract addresses موجود"""
        path_files = set(f.stem.replace("_path_database", "")
                         for f in self.path_db_dir.glob("*_path_database.json"))
        profile_files = set(f.stem.replace("_profile", "")
                            for f in self.profile_dir.glob("*_profile.json"))

        return list(path_files.intersection(profile_files))

    def load_contract(self, contract_address: str) -> Optional[ContractData]:
        """
        خواندن داده‌های کامل یک contract

        Args:
            contract_address: آدرس contract (با یا بدون 0x)

        Returns:
            ContractData object یا None در صورت خطا
        """
        # Normalize address
        if not contract_address.startswith("0x"):
            contract_address = "0x" + contract_address
        contract_address = contract_address.lower()

        # Check cache first
        if contract_address in self.loaded_contracts:
            return self.loaded_contracts[contract_address]

        try:
            # Load path database
            path_file = self.path_db_dir / f"{contract_address}_path_database.json"
            if not path_file.exists():
                logger.error(f"Path database not found for {contract_address}")
                return None

            with open(path_file, 'r', encoding='utf-8') as f:
                path_data = json.load(f)

            # Load profile
            profile_file = self.profile_dir / f"{contract_address}_profile.json"
            if not profile_file.exists():
                logger.error(f"Profile not found for {contract_address}")
                return None

            with open(profile_file, 'r', encoding='utf-8') as f:
                profile_data = json.load(f)

            # Validate structure
            if not self._validate_data_structure(path_data, profile_data):
                return None

            # Create ContractData
            contract = ContractData(
                address=contract_address,
                label=profile_data.get('label', 'unknown'),
                paths=path_data.get('paths', []),
                profile=profile_data,
                total_paths=len(path_data.get('paths', []))
            )

            # Cache it
            self.loaded_contracts[contract_address] = contract

            logger.debug(f"Loaded contract {contract_address[:10]}... "
                         f"({contract.label}) with {contract.total_paths} paths")
            return contract

        except Exception as e:
            logger.error(f"Error loading contract {contract_address}: {str(e)}")
            return None

    def _validate_data_structure(self, path_data: Dict, profile_data: Dict) -> bool:
        """بررسی صحت ساختار داده‌ها"""
        # Check path data
        if 'paths' not in path_data or not isinstance(path_data['paths'], list):
            logger.error("Invalid path data structure")
            return False

        if len(path_data['paths']) == 0:
            logger.warning("No paths found in database")
            return False

        # Check first path has required fields
        sample_path = path_data['paths'][0]
        required_fields = ['path_index', 'basic_info', 'aggregate_features']
        for field in required_fields:
            if field not in sample_path:
                logger.error(f"Missing required field in path: {field}")
                return False

        # Check profile data
        if 'label' not in profile_data:
            logger.error("Missing label in profile")
            return False

        if profile_data['label'] not in ['safe', 'vulnerable']:
            logger.error(f"Invalid label: {profile_data['label']}")
            return False

        return True

    def get_valid_contracts(self, min_paths: int = 5) -> List[Tuple[str, int, str]]:
        """
        لیست contracts معتبر با حداقل تعداد paths

        Args:
            min_paths: حداقل تعداد paths مورد نیاز

        Returns:
            List of tuples: (contract_address, path_count, label)
        """
        # اگر contracts load نشده، ابتدا load کن
        if not hasattr(self, 'contracts'):
            self.contracts = {}

            # خواندن همه path database files
            for file in os.listdir(self.path_db_dir):
                if file.endswith('_path_database.json'):
                    address = file.replace('_path_database.json', '')

                    # Load path database
                    path_file = os.path.join(self.path_db_dir, file)
                    try:
                        with open(path_file, 'r', encoding='utf-8') as f:
                            path_data = json.load(f)
                    except:
                        logger.warning(f"Failed to load {file}")
                        continue

                    # Load contract profile
                    profile_file = os.path.join(self.profile_dir, f'{address}_profile.json')
                    try:
                        with open(profile_file, 'r', encoding='utf-8') as f:
                            profile_data = json.load(f)
                    except:
                        logger.warning(f"No profile for {address}")
                        profile_data = {}

                    # ذخیره contract data
                    self.contracts[address] = {
                        'paths': path_data.get('paths', []),
                        'profile': profile_data
                    }

        # حالا فیلتر کنیم
        valid_contracts = []

        for address, data in self.contracts.items():
            path_count = len(data['paths'])
            label = data['profile'].get('label', 'unknown')

            # فیلتر contracts با paths کم
            if path_count < min_paths:
                logger.debug(f"Skipping contract {address[:10]}... with only {path_count} paths")
                continue

            # فیلتر contracts بدون label
            if label == 'unknown':
                logger.debug(f"Skipping contract {address[:10]}... with unknown label")
                continue

            valid_contracts.append((address, path_count, label))

        logger.info(f"Found {len(valid_contracts)} valid contracts with >= {min_paths} paths")

        # نمایش توزیع
        safe_count = sum(1 for _, _, label in valid_contracts if label == 'safe')
        vuln_count = sum(1 for _, _, label in valid_contracts if label == 'vulnerable')
        logger.info(f"Distribution: {safe_count} safe, {vuln_count} vulnerable")

        return valid_contracts


    def get_contracts_by_label(self, label: str) -> List[str]:
        """برگرداندن contracts با label مشخص"""
        return [addr for addr, _, lbl in self.valid_contracts if lbl == label]

    def get_random_contract(self, label: Optional[str] = None) -> Optional[ContractData]:
        """انتخاب تصادفی یک contract"""
        import random

        if label:
            candidates = self.get_contracts_by_label(label)
        else:
            candidates = [addr for addr, _, _ in self.valid_contracts]

        if not candidates:
            return None

        selected = random.choice(candidates)
        return self.load_contract(selected)

    def get_statistics(self) -> Dict:
        """آمار کلی dataset"""
        total_paths = sum(count for _, count, _ in self.valid_contracts)
        safe_count = sum(1 for _, _, label in self.valid_contracts if label == 'safe')
        vuln_count = len(self.valid_contracts) - safe_count

        return {
            'total_contracts': len(self.valid_contracts),
            'safe_contracts': safe_count,
            'vulnerable_contracts': vuln_count,
            'total_paths': total_paths,
            'avg_paths_per_contract': total_paths / len(self.valid_contracts) if self.valid_contracts else 0,
            'max_paths': max((count for _, count, _ in self.valid_contracts), default=0),
            'min_paths': min((count for _, count, _ in self.valid_contracts), default=0)
        }

    def extract_path_features(self, path: Dict) -> np.ndarray:
        """
        استخراج feature vector از یک path
        Returns: numpy array of features
        """
        features = path.get('aggregate_features', {})

        # Extract numerical features
        feature_vector = [
            features.get('path_length_normalized', 0),
            features.get('require_density', 0),
            features.get('condition_density', 0),
            features.get('keccak_density', 0),
            features.get('mitigation_score', 0),
            features.get('has_any_mitigation', 0),
            features.get('has_strong_mitigation', 0),
            features.get('has_modifier_protection', 0),
            features.get('has_restricted_visibility', 0),
            features.get('has_external_protection', 0),
            features.get('function_require_density', 0),
            features.get('unique_functions_ratio', 0),
            features.get('node_diversity', 0),
            features.get('distance_to_sink', 0),
            features.get('distance_from_source', 0),
            features.get('has_data_flow', 0),
            features.get('contains_loop', 0)
        ]

        # Add encoded features
        source_encoded = features.get('source_type_encoded', [0, 0, 0, 0])
        sink_encoded = features.get('sink_type_encoded', [0, 0, 0, 0])
        feature_vector.extend(source_encoded)
        feature_vector.extend(sink_encoded)

        return np.array(feature_vector, dtype=np.float32)

    def clear_cache(self):
        """پاک کردن cache برای آزاد کردن حافظه"""
        self.loaded_contracts.clear()
        logger.info("Cache cleared")
