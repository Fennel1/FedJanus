# 防御模块
# Defense modules

from .krum import KrumDefense, create_krum_defense, validate_krum_config
from .median import MedianDefense, create_median_defense, validate_median_config
from .custom_defense import (
    CustomDefenseBase,
    FunctionBasedDefense,
    CustomDefenseRegistry,
    register_custom_defense,
    register_function_defense,
    create_custom_defense,
    get_available_custom_defenses,
    validate_custom_defense_config
)

__all__ = [
    'KrumDefense',
    'create_krum_defense', 
    'validate_krum_config',
    'MedianDefense',
    'create_median_defense',
    'validate_median_config',
    'CustomDefenseBase',
    'FunctionBasedDefense',
    'CustomDefenseRegistry',
    'register_custom_defense',
    'register_function_defense',
    'create_custom_defense',
    'get_available_custom_defenses',
    'validate_custom_defense_config'
]