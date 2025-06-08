"""
ファイル管理ユーティリティ
PDFとODTファイルの読み込み機能
"""
import os
from pathlib import Path
from typing import Optional, List, Dict
import logging

# ログ設定
logger = logging.getLogger(__name__)

# プロジェクトのルートパスを設定
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
IPM_KNOWLEDGE_DIR = DATA_DIR / "ipm_knowledge"
RESIDIA_DATA_DIR = DATA_DIR / "residia_data"

# ディレクトリが存在しない場合は作成
IPM_KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
RESIDIA_DATA_DIR.mkdir(parents=True, exist_ok=True)

def get_residia_file_path(residia_type: str) -> Optional[Path]:
    """
    レジディアタイプに対応するODTファイルのパスを取得
    
    Args:
        residia_type: レジディアタイプ名（例：「背信」）
    
    Returns:
        ファイルパス（存在しない場合はNone）
    """
    file_name = f"{residia_type}のレジディア.odt"
    file_path = RESIDIA_DATA_DIR / file_name
    
    if file_path.exists():
        return file_path
    else:
        logger.warning(f"レジディアファイルが見つかりません: {file_name}")
        return None

def get_all_residia_files() -> Dict[str, Path]:
    """
    すべてのレジディアファイルのパスを取得
    
    Returns:
        {レジディアタイプ: ファイルパス}の辞書
    """
    residia_types = ["背信", "不道徳", "無欲", "哀感", "苛烈", "切断"]
    files = {}
    
    for residia_type in residia_types:
        file_path = get_residia_file_path(residia_type)
        if file_path:
            files[residia_type] = file_path
    
    return files

def get_ipm_knowledge_files() -> List[Path]:
    """
    IPM知識PDFファイルのパスリストを取得
    
    Returns:
        PDFファイルパスのリスト
    """
    pdf_files = []
    
    if IPM_KNOWLEDGE_DIR.exists():
        pdf_files = list(IPM_KNOWLEDGE_DIR.glob("*.pdf"))
        if not pdf_files:
            logger.warning("IPM知識PDFファイルが見つかりません")
    
    return pdf_files

def check_data_files_status() -> Dict[str, any]:
    """
    データファイルの状態を確認
    
    Returns:
        ファイル存在状況の辞書
    """
    status = {
        "data_dir_exists": DATA_DIR.exists(),
        "ipm_knowledge_dir": {
            "exists": IPM_KNOWLEDGE_DIR.exists(),
            "pdf_count": len(get_ipm_knowledge_files()),
            "files": [f.name for f in get_ipm_knowledge_files()]
        },
        "residia_data_dir": {
            "exists": RESIDIA_DATA_DIR.exists(),
            "files": {}
        }
    }
    
    # レジディアファイルの存在確認
    residia_files = get_all_residia_files()
    for residia_type, file_path in residia_files.items():
        status["residia_data_dir"]["files"][residia_type] = {
            "exists": True,
            "path": str(file_path),
            "size": file_path.stat().st_size if file_path.exists() else 0
        }
    
    # 存在しないレジディアファイルも記録
    all_types = ["背信", "不道徳", "無欲", "哀感", "苛烈", "切断"]
    for residia_type in all_types:
        if residia_type not in residia_files:
            status["residia_data_dir"]["files"][residia_type] = {
                "exists": False,
                "path": str(RESIDIA_DATA_DIR / f"{residia_type}のレジディア.odt"),
                "size": 0
            }
    
    return status

# AIがファイルを参照する際のインターフェース
def get_file_content_for_ai(file_type: str, specific_type: Optional[str] = None) -> Dict[str, str]:
    """
    AIが参照するためのファイルパス情報を提供
    
    Args:
        file_type: "ipm" または "residia"
        specific_type: レジディアの場合は具体的なタイプ名
    
    Returns:
        AIに渡すためのファイルパス情報
    """
    result = {
        "status": "success",
        "file_paths": [],
        "instructions": ""
    }
    
    if file_type == "ipm":
        pdf_files = get_ipm_knowledge_files()
        if pdf_files:
            result["file_paths"] = [str(f) for f in pdf_files]
            result["instructions"] = "これらのPDFファイルを参照して、IPMの知識に基づいて回答してください。"
        else:
            result["status"] = "no_files"
            result["instructions"] = "IPM知識PDFファイルが見つかりません。"
    
    elif file_type == "residia":
        if specific_type:
            file_path = get_residia_file_path(specific_type)
            if file_path:
                result["file_paths"] = [str(file_path)]
                result["instructions"] = f"{specific_type}のレジディアファイルを参照して分析してください。"
            else:
                result["status"] = "file_not_found"
                result["instructions"] = f"{specific_type}のレジディアファイルが見つかりません。"
        else:
            residia_files = get_all_residia_files()
            if residia_files:
                result["file_paths"] = [str(f) for f in residia_files.values()]
                result["instructions"] = "すべてのレジディアファイルを参照して、最適なタイプを判定してください。"
            else:
                result["status"] = "no_files"
                result["instructions"] = "レジディアファイルが見つかりません。"
    
    return result

if __name__ == "__main__":
    # ファイル状態の確認テスト
    import json
    status = check_data_files_status()
    print("データファイルの状態:")
    print(json.dumps(status, indent=2, ensure_ascii=False))