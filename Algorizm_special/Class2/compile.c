#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdarg.h>

/* 定数定義 */
#define MAX_TOKEN_LEN 256
#define MAX_TOKENS 10000
#define MAX_SYMBOL_TABLE 1000
#define MAX_CODE_SIZE 10000
#define MAX_LABEL_LEN 64
#define MAX_STR_CONST 1000

/* トークンの種類 */
typedef enum {
    TOKEN_IDENTIFIER,
    TOKEN_NUMBER,
    TOKEN_STRING,
    TOKEN_OPERATOR,
    TOKEN_KEYWORD,
    TOKEN_SEMICOLON,
    TOKEN_COMMA,
    TOKEN_BRACKET_OPEN,
    TOKEN_BRACKET_CLOSE,
    TOKEN_BRACE_OPEN,
    TOKEN_BRACE_CLOSE,
    TOKEN_PAREN_OPEN,
    TOKEN_PAREN_CLOSE,
    TOKEN_EOF
} TokenType;

/* トークン構造体 */
typedef struct {
    TokenType type;
    char value[MAX_TOKEN_LEN];
    int line_number;
    int column;
} Token;

/* シンボルの種類 */
typedef enum {
    SYMBOL_VARIABLE,
    SYMBOL_ARRAY,
    SYMBOL_FUNCTION,
    SYMBOL_PARAMETER,
    SYMBOL_TEMP
} SymbolType;

/* シンボルテーブル構造体 */
typedef struct {
    char name[MAX_TOKEN_LEN];
    int address;         // メモリアドレスまたは関数ラベル
    SymbolType type;     // シンボルの種類
    int size;            // 配列の場合はサイズ、それ以外は1
    int scope_level;     // スコープレベル（グローバル=0）
} Symbol;

/* ダンプするための命令構造体 */
typedef struct {
    char opcode[16];
    char operand[MAX_TOKEN_LEN];
    int is_label;        // ラベル行かどうか
} Instruction;

/* グローバル変数 */
Token tokens[MAX_TOKENS];
int token_count = 0;
int current_token = 0;

Symbol symbol_table[MAX_SYMBOL_TABLE];
int symbol_count = 0;

Instruction ram_code[MAX_CODE_SIZE];
int ram_code_line = 0;

int temp_var_count = 0;
int label_count = 0;
int current_scope = 0;
int data_address = 1;    // データ領域の開始アドレス

char string_constants[MAX_STR_CONST][MAX_TOKEN_LEN];
int string_count = 0;

/* キーワードのリスト */
const char* keywords[] = {
    "int", "if", "else", "while", "for", "return", "void", "main",
    "char", "break", "continue", "do", "switch", "case", "default",
    NULL
};

/* 関数プロトタイプ */
void error(const char* format, ...);
void tokenize(const char* source);
int is_keyword(const char* str);
int add_symbol(const char* name, SymbolType type, int size);
int find_symbol(const char* name);
int find_symbol_in_scope(const char* name, int scope);
char* gen_temp_var();
char* gen_label();
void emit_instruction(const char* opcode, const char* operand, int is_label);
void emit(const char* format, ...);
void consume();
Token* current();
Token* peek(int ahead);
int match(TokenType type);
void expect(TokenType type);
void expression();
void statement();
void function_definition();
void program();

/* エラー報告 */
void error(const char* format, ...) {
    va_list args;
    va_start(args, format);
    
    fprintf(stderr, "Error: ");
    vfprintf(stderr, format, args);
    
    if (current_token < token_count) {
        fprintf(stderr, " at line %d, column %d\n", 
                tokens[current_token].line_number, 
                tokens[current_token].column);
    } else {
        fprintf(stderr, " at end of file\n");
    }
    
    va_end(args);
    exit(1);
}

/* キーワードかどうかチェック */
int is_keyword(const char* str) {
    for (int i = 0; keywords[i] != NULL; i++) {
        if (strcmp(str, keywords[i]) == 0) {
            return 1;
        }
    }
    return 0;
}

/* 字句解析 */
void tokenize(const char* source) {
    int i = 0;
    int line = 1;
    int column = 1;
    
    while (source[i] != '\0') {
        // 空白をスキップ
        while (isspace(source[i])) {
            if (source[i] == '\n') {
                line++;
                column = 1;
            } else {
                column++;
            }
            i++;
        }
        
        if (source[i] == '\0') break;
        
        // コメントをスキップ
        if (source[i] == '/' && source[i+1] == '/') {
            i += 2;
            column += 2;
            while (source[i] != '\n' && source[i] != '\0') {
                i++;
                column++;
            }
            continue;
        }
        
        if (source[i] == '/' && source[i+1] == '*') {
            int comment_line = line;
            int comment_col = column;
            i += 2;
            column += 2;
            while (!(source[i] == '*' && source[i+1] == '/')) {
                if (source[i] == '\n') {
                    line++;
                    column = 1;
                } else {
                    column++;
                }
                if (source[i] == '\0') {
                    fprintf(stderr, "Error: unclosed comment at line %d, column %d\n", 
                            comment_line, comment_col);
                    exit(1);
                }
                i++;
            }
            i += 2;
            column += 2;
            continue;
        }
        
        // 識別子またはキーワード
        if (isalpha(source[i]) || source[i] == '_') {
            char buffer[MAX_TOKEN_LEN];
            int buf_idx = 0;
            int start_col = column;
            
            while (isalnum(source[i]) || source[i] == '_') {
                buffer[buf_idx++] = source[i++];
                column++;
            }
            buffer[buf_idx] = '\0';
            
            tokens[token_count].type = is_keyword(buffer) ? TOKEN_KEYWORD : TOKEN_IDENTIFIER;
            strcpy(tokens[token_count].value, buffer);
            tokens[token_count].line_number = line;
            tokens[token_count].column = start_col;
            token_count++;
            continue;
        }
        
        // 数値
        if (isdigit(source[i])) {
            char buffer[MAX_TOKEN_LEN];
            int buf_idx = 0;
            int start_col = column;
            
            while (isdigit(source[i])) {
                buffer[buf_idx++] = source[i++];
                column++;
            }
            buffer[buf_idx] = '\0';
            
            tokens[token_count].type = TOKEN_NUMBER;
            strcpy(tokens[token_count].value, buffer);
            tokens[token_count].line_number = line;
            tokens[token_count].column = start_col;
            token_count++;
            continue;
        }
        
        // 文字列リテラル
        if (source[i] == '"') {
            char buffer[MAX_TOKEN_LEN];
            int buf_idx = 0;
            int start_col = column;
            
            buffer[buf_idx++] = source[i++]; // 開始の "
            column++;
            
            while (source[i] != '"' && source[i] != '\0') {
                if (source[i] == '\n') {
                    fprintf(stderr, "Error: newline in string literal at line %d\n", line);
                    exit(1);
                }
                
                if (source[i] == '\\' && source[i+1] != '\0') {
                    buffer[buf_idx++] = source[i++];
                    column++;
                }
                buffer[buf_idx++] = source[i++];
                column++;
            }
            
            if (source[i] == '"') {
                buffer[buf_idx++] = source[i++]; // 終了の "
                column++;
            } else {
                fprintf(stderr, "Error: unterminated string at line %d, column %d\n", 
                        line, start_col);
                exit(1);
            }
            
            buffer[buf_idx] = '\0';
            
            tokens[token_count].type = TOKEN_STRING;
            strcpy(tokens[token_count].value, buffer);
            tokens[token_count].line_number = line;
            tokens[token_count].column = start_col;
            token_count++;
            
            // 文字列定数を保存
            strcpy(string_constants[string_count++], buffer);
            continue;
        }
        
        // 演算子と記号
        int start_col = column;
        switch (source[i]) {
            case ';':
                tokens[token_count].type = TOKEN_SEMICOLON;
                tokens[token_count].value[0] = ';';
                tokens[token_count].value[1] = '\0';
                tokens[token_count].line_number = line;
                tokens[token_count].column = column;
                token_count++;
                i++;
                column++;
                break;
            case ',':
                tokens[token_count].type = TOKEN_COMMA;
                tokens[token_count].value[0] = ',';
                tokens[token_count].value[1] = '\0';
                tokens[token_count].line_number = line;
                tokens[token_count].column = column;
                token_count++;
                i++;
                column++;
                break;
            case '{':
                tokens[token_count].type = TOKEN_BRACE_OPEN;
                tokens[token_count].value[0] = '{';
                tokens[token_count].value[1] = '\0';
                tokens[token_count].line_number = line;
                tokens[token_count].column = column;
                token_count++;
                i++;
                column++;
                break;
            case '}':
                tokens[token_count].type = TOKEN_BRACE_CLOSE;
                tokens[token_count].value[0] = '}';
                tokens[token_count].value[1] = '\0';
                tokens[token_count].line_number = line;
                tokens[token_count].column = column;
                token_count++;
                i++;
                column++;
                break;
            case '(':
                tokens[token_count].type = TOKEN_PAREN_OPEN;
                tokens[token_count].value[0] = '(';
                tokens[token_count].value[1] = '\0';
                tokens[token_count].line_number = line;
                tokens[token_count].column = column;
                token_count++;
                i++;
                column++;
                break;
            case ')':
                tokens[token_count].type = TOKEN_PAREN_CLOSE;
                tokens[token_count].value[0] = ')';
                tokens[token_count].value[1] = '\0';
                tokens[token_count].line_number = line;
                tokens[token_count].column = column;
                token_count++;
                i++;
                column++;
                break;
            case '[':
                tokens[token_count].type = TOKEN_BRACKET_OPEN;
                tokens[token_count].value[0] = '[';
                tokens[token_count].value[1] = '\0';
                tokens[token_count].line_number = line;
                tokens[token_count].column = column;
                token_count++;
                i++;
                column++;
                break;
            case ']':
                tokens[token_count].type = TOKEN_BRACKET_CLOSE;
                tokens[token_count].value[0] = ']';
                tokens[token_count].value[1] = '\0';
                tokens[token_count].line_number = line;
                tokens[token_count].column = column;
                token_count++;
                i++;
                column++;
                break;
            default:
                // 複合演算子や単一演算子を処理
                if (strchr("+-*/%=<>!&|^~?:", source[i])) {
                    char buffer[3] = {0};
                    buffer[0] = source[i++];
                    column++;
                    
                    // 2文字の演算子をチェック
                    if ((buffer[0] == '+' && source[i] == '+') ||
                        (buffer[0] == '-' && source[i] == '-') ||
                        (buffer[0] == '=' && source[i] == '=') ||
                        (buffer[0] == '!' && source[i] == '=') ||
                        (buffer[0] == '<' && source[i] == '=') ||
                        (buffer[0] == '>' && source[i] == '=') ||
                        (buffer[0] == '&' && source[i] == '&') ||
                        (buffer[0] == '|' && source[i] == '|') ||
                        (buffer[0] == '+' && source[i] == '=') ||
                        (buffer[0] == '-' && source[i] == '=') ||
                        (buffer[0] == '*' && source[i] == '=') ||
                        (buffer[0] == '/' && source[i] == '=') ||
                        (buffer[0] == '%' && source[i] == '=') ||
                        (buffer[0] == '<' && source[i] == '<') ||
                        (buffer[0] == '>' && source[i] == '>')) {
                        buffer[1] = source[i++];
                        column++;
                    }
                    
                    tokens[token_count].type = TOKEN_OPERATOR;
                    strcpy(tokens[token_count].value, buffer);
                    tokens[token_count].line_number = line;
                    tokens[token_count].column = start_col;
                    token_count++;
                } else {
                    // 未知の文字はスキップ
                    fprintf(stderr, "Warning: ignoring character '%c' at line %d, column %d\n", 
                            source[i], line, column);
                    i++;
                    column++;
                }
                break;
        }
    }
    
    // EOFトークンを追加
    tokens[token_count].type = TOKEN_EOF;
    strcpy(tokens[token_count].value, "EOF");
    tokens[token_count].line_number = line;
    tokens[token_count].column = column;
    token_count++;
}

/* シンボルテーブル操作関数 */
int add_symbol(const char* name, SymbolType type, int size) {
    // 現在のスコープで既に存在するか確認
    for (int i = 0; i < symbol_count; i++) {
        if (symbol_table[i].scope_level == current_scope && 
            strcmp(symbol_table[i].name, name) == 0) {
            return symbol_table[i].address;
        }
    }
    
    // 新しいシンボルを追加
    strcpy(symbol_table[symbol_count].name, name);
    symbol_table[symbol_count].type = type;
    symbol_table[symbol_count].size = size;
    symbol_table[symbol_count].scope_level = current_scope;
    
    // アドレスを割り当て
    if (type == SYMBOL_FUNCTION) {
        // 関数の場合はアドレス不要
        symbol_table[symbol_count].address = 0;
    } else {
        symbol_table[symbol_count].address = data_address;
        data_address += size; // サイズ分アドレスを進める
    }
    
    return symbol_table[symbol_count++].address;
}

/* シンボルを検索 */
int find_symbol(const char* name) {
    // 最も内側のスコープから探す
    for (int scope = current_scope; scope >= 0; scope--) {
        for (int i = 0; i < symbol_count; i++) {
            if (symbol_table[i].scope_level == scope && 
                strcmp(symbol_table[i].name, name) == 0) {
                return i;
            }
        }
    }
    return -1; // 見つからない場合
}

/* 指定されたスコープ内でシンボルを検索 */
int find_symbol_in_scope(const char* name, int scope) {
    for (int i = 0; i < symbol_count; i++) {
        if (symbol_table[i].scope_level == scope && 
            strcmp(symbol_table[i].name, name) == 0) {
            return i;
        }
    }
    return -1;
}

/* 一時変数を生成 */
char* gen_temp_var() {
    static char temp_name[MAX_TOKEN_LEN];
    sprintf(temp_name, "t%d", temp_var_count++);
    add_symbol(temp_name, SYMBOL_TEMP, 1);
    return temp_name;
}

/* ラベルを生成 */
char* gen_label() {
    static char label[MAX_LABEL_LEN];
    sprintf(label, "L%d", label_count++);
    return label;
}

/* RAM命令を出力（構造体に保存） */
void emit_instruction(const char* opcode, const char* operand, int is_label) {
    strcpy(ram_code[ram_code_line].opcode, opcode);
    strcpy(ram_code[ram_code_line].operand, operand);
    ram_code[ram_code_line].is_label = is_label;
    ram_code_line++;
}

/* 書式付きRAM命令を出力 */
void emit(const char* format, ...) {
    char buffer[MAX_TOKEN_LEN * 2];
    va_list args;
    va_start(args, format);
    vsprintf(buffer, format, args);
    va_end(args);
    
    // ラベルかどうか確認
    int is_label = 0;
    char opcode[MAX_TOKEN_LEN];
    char operand[MAX_TOKEN_LEN] = "";
    
    if (strchr(buffer, ':') != NULL) {
        // ラベル行
        is_label = 1;
        strcpy(opcode, buffer);
    } else {
        // 通常の命令行
        char* space = strchr(buffer, ' ');
        if (space) {
            *space = '\0';
            strcpy(opcode, buffer);
            strcpy(operand, space + 1);
        } else {
            strcpy(opcode, buffer);
        }
    }
    
    emit_instruction(opcode, operand, is_label);
}

/* 次のトークンを消費 */
void consume() {
    if (current_token < token_count) {
        current_token++;
    }
}

/* 現在のトークンを取得 */
Token* current() {
    return &tokens[current_token];
}

/* 先読み */
Token* peek(int ahead) {
    if (current_token + ahead < token_count) {
        return &tokens[current_token + ahead];
    }
    return current();
}

/* トークンタイプが一致するか確認し、一致すれば消費 */
int match(TokenType type) {
    if (current()->type == type) {
        consume();
        return 1;
    }
    return 0;
}

/* 期待するトークンタイプでなければエラー */
void expect(TokenType type) {
    if (!match(type)) {
        error("expected %d, got %d (%s)", type, current()->type, current()->value);
    }
}

/* 以下、構文解析と意味解析の関数 */

/* 因子の解析 */
void factor() {
    if (match(TOKEN_NUMBER)) {
        // 数値リテラル
        emit("LOAD =%s", tokens[current_token-1].value);
    } else if (match(TOKEN_IDENTIFIER)) {
        // 変数
        char var_name[MAX_TOKEN_LEN];
        strcpy(var_name, tokens[current_token-1].value);
        
        // シンボルテーブルで検索
        int sym_idx = find_symbol(var_name);
        if (sym_idx == -1) {
            error("undefined variable '%s'", var_name);
        }
        
        // 配列アクセスの場合
        if (match(TOKEN_BRACKET_OPEN)) {
            // インデックスが入ったr0を退避
            char* temp = gen_temp_var();
            emit("STORE %s", temp);
            
            // インデックス式を評価（結果はr0に）
            expression();
            expect(TOKEN_BRACKET_CLOSE);
            
            // 配列のベースアドレスを加算
            emit("ADD =%d", symbol_table[sym_idx].address - 1);
            emit("STORE r1");  // 計算されたアドレスをr1に保存
            emit("LOAD *r1");  // r1が指すアドレスの内容をr0にロード
            
            // 元のr0を復元
            char* temp2 = gen_temp_var();
            emit("STORE %s", temp2);
            emit("LOAD %s", temp);
        } else {
            // 通常の変数アクセス
            emit("LOAD %s", var_name);
        }
    } else if (match(TOKEN_PAREN_OPEN)) {
        // 括弧で囲まれた式
        expression();
        expect(TOKEN_PAREN_CLOSE);
    } else if (match(TOKEN_OPERATOR) && (strcmp(tokens[current_token-1].value, "-") == 0 ||
                                        strcmp(tokens[current_token-1].value, "!") == 0)) {
        // 単項演算子
        char op[MAX_TOKEN_LEN];
        strcpy(op, tokens[current_token-1].value);
        
        factor();
        
        if (strcmp(op, "-") == 0) {
            // 単項マイナス
            emit("STORE r1");
            emit("LOAD =0");
            emit("SUB r1");
        } else if (strcmp(op, "!") == 0) {
            // 論理否定
            char* label_true = gen_label();
            char* label_end = gen_label();
            
            emit("JZERO %s", label_true);
            emit("LOAD =0");  // false
            emit("JUMP %s", label_end);
            emit("%s:", label_true);
            emit("LOAD =1");  // true
            emit("%s:", label_end);
        }
    } else {
        error("unexpected token in factor: %s", current()->value);
    }
}

/* 項の解析 */
void term() {
    factor();
    
    while (current()->type == TOKEN_OPERATOR && 
           (strcmp(current()->value, "*") == 0 || 
            strcmp(current()->value, "/") == 0 || 
            strcmp(current()->value, "%") == 0)) {
        
        char op[MAX_TOKEN_LEN];
        strcpy(op, current()->value);
        consume(); // 演算子を消費
        
        char* temp = gen_temp_var();
        emit("STORE %s", temp); // 左辺の結果を保存
        
        factor(); // 右辺を評価
        
        // 演算を実行
        if (strcmp(op, "*") == 0) {
            char* right = gen_temp_var();
            emit("STORE %s", right);
            emit("LOAD %s", temp);
            emit("MULT %s", right);
        } else if (strcmp(op, "/") == 0) {
            char* right = gen_temp_var();
            emit("STORE %s", right);
            emit("LOAD %s", temp);
            emit("DIV %s", right);
        } else if (strcmp(op, "%") == 0) {
            // RAMには直接のモジュロ演算子がないため、実装
            char* right = gen_temp_var();
            char* quot = gen_temp_var();
            
            emit("STORE %s", right);
            emit("LOAD %s", temp);
            emit("DIV %s", right);    // 除算
            emit("STORE %s", quot);   // 商を保存
            emit("MULT %s", right);   // 商 * 右辺
            emit("STORE %s", right);  // 中間結果を保存
            emit("LOAD %s", temp);    // 左辺を再ロード
            emit("SUB %s", right);    // 左辺 - (商 * 右辺) = 剰余
        }
    }
}

/* 加減の解析 */
void additive_expression() {
    term();
    
    while (current()->type == TOKEN_OPERATOR && 
           (strcmp(current()->value, "+") == 0 || 
            strcmp(current()->value, "-") == 0)) {
        
        char op[MAX_TOKEN_LEN];
        strcpy(op, current()->value);
        consume(); // 演算子を消費
        
        char* temp = gen_temp_var();
        emit("STORE %s", temp); // 左辺の結果を保存
        
        term(); // 右辺を評価
        
        // 演算を実行
        char* right = gen_temp_var();
        emit("STORE %s", right);
        emit("LOAD %s", temp);
        
        if (strcmp(op, "+") == 0) {
            emit("ADD %s", right);
        } else if (strcmp(op, "-") == 0) {
            emit("SUB %s", right);
        }
    }
}

/* 比較式の解析 */
void relational_expression() {
    additive_expression();
    
    if (current()->type == TOKEN_OPERATOR && 
        (strcmp(current()->value, "<") == 0 || 
         strcmp(current()->value, ">") == 0 || 
         strcmp(current()->value, "<=") == 0 || 
         strcmp(current()->value, ">=") == 0)) {
        
        char op[MAX_TOKEN_LEN];
        strcpy(op, current()->value);
        consume(); // 演算子を消費
        
        char* left = gen_temp_var();
        emit("STORE %s", left); // 左辺の結果を保存
        
        additive_expression(); // 右辺を評価
        
        char* right = gen_temp_var();
        emit("STORE %s", right);
        
        char* true_label = gen_label();
        char* false_label = gen_label();
        char* end_label = gen_label();
        
        // 左辺と右辺を比較
        emit("LOAD %s", left);
        emit("SUB %s", right);  // 左辺 - 右辺
        
        if (strcmp(op, "<") == 0) {
            // 左辺 < 右辺 は (左辺 - 右辺) < 0
            emit("JGTZ %s", false_label);
            emit("JZERO %s", false_label);
            emit("JUMP %s", true_label);
        } else if (strcmp(op, ">") == 0) {
            // 左辺 > 右辺 は (左辺 - 右辺) > 0
            emit("JGTZ %s", true_label);
            emit("JUMP %s", false_label);
        } else if (strcmp(op, "<=") == 0) {
            // 左辺 <= 右辺 は !(左辺 > 右辺)
            emit("JGTZ %s", false_label);
            emit("JUMP %s", true_label);
        } else if (strcmp(op, ">=") == 0) {
            // 左辺 >= 右辺 は !(左辺 < 右辺)
            emit("JGTZ %s", true_label);
            emit("JZERO %s", true_label);
            emit("JUMP %s", false_label);
        }
        
        emit("%s:", true_label);
        emit("LOAD =1");  // true
        emit("JUMP %s", end_label);
        emit("%s:", false_label);
        emit("LOAD =0");  // false
        emit("%s:", end_label);
    }
}

/* 等価式の解析 */
void equality_expression() {
    relational_expression();
    
    if (current()->type == TOKEN_OPERATOR && 
        (strcmp(current()->value, "==") == 0 || 
         strcmp(current()->value, "!=") == 0)) {
        
        char op[MAX_TOKEN_LEN];
        strcpy(op, current()->value);
        consume(); // 演算子を消費
        
        char* left = gen_temp_var();
        emit("STORE %s", left); // 左辺の結果を保存
        
        relational_expression(); // 右辺を評価
        
        char* right = gen_temp_var();
        emit("STORE %s", right);
        
        char* true_label = gen_label();
        char* false_label = gen_label();
        char* end_label = gen_label();
        
        // 左辺と右辺を比較
        emit("LOAD %s", left);
        emit("SUB %s", right);  // 左辺 - 右辺
        
        if (strcmp(op, "==") == 0) {
            // 左辺 == 右辺 は (左辺 - 右辺) == 0
            emit("JZERO %s", true_label);
            emit("JUMP %s", false_label);
        } else if (strcmp(op, "!=") == 0) {
            // 左辺 != 右辺 は (左辺 - 右辺) != 0
            emit("JZERO %s", false_label);
            emit("JUMP %s", true_label);
        }
        emit("%s:", true_label);
        emit("LOAD =1");  // true
        emit("JUMP %s", end_label);
        emit("%s:", false_label);
        emit("LOAD =0");  // false
        emit("%s:", end_label);
    }
}
/* 論理式の解析 */
