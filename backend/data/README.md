# データファイル配置ガイド

このフォルダには、IPMカウンセリングアプリで使用するデータファイルを配置します。

## フォルダ構造
data/
├── ipm_knowledge/IPM.pdf     # IPM知識のPDFファイル
│   └── （300ページのPDFファイルをここに配置）
└── residia_data/      # レジディアのODTファイル
├── 背信のレジディア.odt
├── 不道徳のレジディア.odt
├── 無欲のレジディア.odt
├── 哀感のレジディア.odt
├── 苛烈のレジディア.odt
└── 切断のレジディア.odt

## ファイル配置手順

1. IPM知識PDFファイル
   - `ipm_knowledge`フォルダにPDFファイルをコピー
   - ファイル名は任意（日本語OK）

2. レジディアODTファイル
   - `residia_data`フォルダに6つのODTファイルをコピー
   - ファイル名は上記の通り正確に設定すること

## 注意事項

- これらのファイルはGitには含まれません（.gitignoreで除外）
- ファイルサイズが大きい場合は、配置に時間がかかることがあります
- ファイル名は大文字・小文字を区別します