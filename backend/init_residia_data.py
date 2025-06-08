"""
レジディアマスターデータの初期投入スクリプト
"""
from app.database import SessionLocal
from app.models.residia import ResidiaType

def init_residia_types():
    """レジディアタイプのマスターデータを投入"""
    db = SessionLocal()
    
    residia_types = [
        {
            "name": "背信",
            "description": "信頼していた人からの裏切りや、約束の破棄による心の傷",
            "keywords": ["裏切り", "信頼", "約束", "期待", "失望"],
            "file_path": "residia_data/背信のレジディア.odt"
        },
        {
            "name": "不道徳",
            "description": "倫理的・道徳的な葛藤や罪悪感による内的な傷",
            "keywords": ["罪悪感", "倫理", "道徳", "良心", "後悔"],
            "file_path": "residia_data/不道徳のレジディア.odt"
        },
        {
            "name": "無欲",
            "description": "欲望や願望を抑圧することで生じた空虚感",
            "keywords": ["抑圧", "欲望", "願望", "諦め", "空虚"],
            "file_path": "residia_data/無欲のレジディア.odt"
        },
        {
            "name": "哀感",
            "description": "深い悲しみや喪失感による心の傷",
            "keywords": ["悲しみ", "喪失", "別離", "孤独", "寂しさ"],
            "file_path": "residia_data/哀感のレジディア.odt"
        },
        {
            "name": "苛烈",
            "description": "厳しい環境や過酷な体験による心の傷",
            "keywords": ["厳しさ", "過酷", "プレッシャー", "要求", "完璧主義"],
            "file_path": "residia_data/苛烈のレジディア.odt"
        },
        {
            "name": "切断",
            "description": "重要な関係や繋がりの断絶による心の傷",
            "keywords": ["断絶", "分離", "孤立", "疎外", "拒絶"],
            "file_path": "residia_data/切断のレジディア.odt"
        }
    ]
    
    try:
        for residia_data in residia_types:
            # 既存チェック
            existing = db.query(ResidiaType).filter(
                ResidiaType.name == residia_data["name"]
            ).first()
            
            if not existing:
                residia_type = ResidiaType(**residia_data)
                db.add(residia_type)
                print(f"追加: {residia_data['name']}のレジディア")
            else:
                print(f"既存: {residia_data['name']}のレジディア")
        
        db.commit()
        print("\nレジディアタイプの初期化が完了しました！")
        
    except Exception as e:
        print(f"エラー: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    init_residia_types()