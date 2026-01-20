# M1_Class
Repository for managing university courses

# ディレクトリ構成

```tree
M1_Class
├── README.md ....説明書
├── .gitignore ....GitHubにpushしないファイル
├── /[講義ディレクトリ] ....講義ごとのディレクトリ
│   ├── README.md ....講義の説明書
│   ├── /[講義資料] ....講義資料のディレクトリ
```

## リポジトリのクローン

```zsh
git clone git@github.com:thomas0124/M1_Class.git
```

## プログラムの実行 (Makefileを使用)

このリポジトリには、各講義ディレクトリ内のスクリプトを自動実行するための `Makefile` が含まれています。これにより、Python、JavaScript、C、C++、およびシェルスクリプトファイルを一括で実行できます。

**使用方法:**

1.  **依存関係のインストール:**
    プロジェクト内の `requirements.txt` ファイルに基づいてPythonの依存ライブラリをインストールします。
    ```bash
    make install
    ```

2.  **すべてのプログラムの実行:**
    `AI_special`、`Algorizm_special`、`NLP_special` ディレクトリ内のすべての実行可能ファイルを順に実行します。
    ```bash
    make      # または
    make all
    ```

3.  **カテゴリ別のプログラム実行:**
    特定のカテゴリのプログラムのみを実行する場合に使用します。
    *   AI関連のPythonスクリプトを実行:
        ```bash
        make run_ai
        ```
    *   アルゴリズム関連のJavaScript, C, C++スクリプトを実行:
        ```bash
        make run_algo
        ```
    *   NLP関連のPython, シェルスクリプトを実行:
        ```bash
        make run_nlp
        ```

4.  **コンパイル済みファイルのクリーンアップ:**
    C/C++プログラムのコンパイルによって生成された実行可能ファイルを削除します。
    ```bash
    make clean
    ```

**注意点:**

*   **Jupyter Notebook (.ipynb):** Jupyter Notebookファイルは、対話的な実行が主な用途であるため、`Makefile` による自動実行の対象外としています。
*   **必要なツール:** プログラムを実行するためには、以下のツールがシステムにインストールされている必要があります。
    *   Pythonスクリプト (`.py`): `python3`
    *   JavaScriptスクリプト (`.js`): `node`
    *   C言語スクリプト (`.c`): `gcc`
    *   C++スクリプト (`.cpp`): `g++`
    *   シェルスクリプト (`.sh`): `bash`



