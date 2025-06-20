from flask import Flask, render_template, request, redirect, flash, url_for

### インポート
import torch
import pandas as pd
import re
import spacy
import random
import nltk
from nltk.stem import WordNetLemmatizer
from word_forms.word_forms import get_word_forms
# import astは消しておく
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from markupsafe import Markup
import os

app = Flask(__name__)

# モデル＆依存ライブラリのロード（起動時に一度だけ）
nltk.download("wordnet")
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])  # 軽量化のためNERやparserを無効に
lemmatizer = WordNetLemmatizer()
# 以下、元あったコード
# Natural language tools
# nltk.download("wordnet")
# nlp = spacy.load("en_core_web_sm")
# lemmatizer = WordNetLemmatizer()


# GPUを使用可能か確認し、使用する
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# renderの無料枠ではGPUは使えず結局CPUになるので、この記述でいく
device = torch.device("cpu")
model_name = "sshleifer/tiny-gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
model.config.pad_token_id = tokenizer.pad_token_id


### CSVファイル
df = pd.read_csv("./static/csv/gutenberg_all_books.csv")

CSV_MAP = {
    "A1": "./static/csv/modified_A1.csv",
    "A2": "./static/csv/modified_A2.csv",
    "B1": "./static/csv/modified_B1.csv",
    "B2": "./static/csv/modified_B2.csv",
}
### 関数の作成


# 文章生成
def generate_text(prompt, max_new_tokens=50, temperature=0.7):
    """
    テキスト生成の関数
    prompt: 入力テキスト（開始文）
    max_length: 生成するテキストの最大長さ
    temperature: 生成するテキストの創造性（高いほどランダム性が増す）
    """
    # 入力テキストをトークン化
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    # モデルでテキスト生成
    outputs = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        num_return_sequences=1,  # 生成するテキストの数
        no_repeat_ngram_size=2,  # 重複するフレーズを避ける
        top_p=0.95,  # 上位p%の確率で生成する
        top_k=50,  # トップk個の候補からサンプリング
        do_sample=True,  # サンプリングを使用
    )
    # 出力をデコードしてテキストに変換
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text
# ダミー作成
# 品詞を揃えてダミーを作成する
def dummy_samepos(HATENA_Block, HATENA_POS, level):
    # 対応するレベルのCSVファイルを読み込む
    df_words_all = pd.read_csv(CSV_MAP[level], encoding="utf-8")

    # HATENA_Block(正解の選択肢)と同じ品詞のものを作成する
    df_words_restricted = df_words_all[df_words_all["pos"] == HATENA_POS]

    # HATENA_Blockを除く(被ると困るので)
    # 修正版
    df_words_restricted = df_words_restricted[
        df_words_restricted.iloc[:, 0].str.lower() != str(HATENA_Block).lower()
    ]

    # df_words_restrictedが空でないか確認
    if df_words_restricted.empty:
        return "DON'T select this."

    # df_words_restrictedからダミーとするものを選ぶ
    dummy = df_words_restricted.iloc[random.randint(0, len(df_words_restricted) - 1)]

    print_memory_usage()

    return dummy[0]

# 品詞に関係なくダミーを作成する
def dummy_normal(HATENA_Block, level):
    # 対応するレベルのCSVファイルを読み込む
    df_words_all = pd.read_csv(CSV_MAP[level], encoding="utf-8")

    # HATENA_Blockを除く
    # 修正版
    df_words_restricted = df_words_all[
        df_words_all.iloc[:, 0].str.lower() != str(HATENA_Block).lower()
    ]

    # df_words_restrictedが空でないか確認
    if df_words_restricted.empty:
        return "DON'T select this."

    # df_words_restrictedからダミーとするものを選ぶ
    dummy = df_words_restricted.iloc[random.randint(0, len(df_words_restricted) - 1)]

    return dummy[0]

# 動詞の型を判定する
def detect_verb_form(word):
    """
    単語がどの活用形 (原形, 三単現, 過去形, ing) に該当するかを判定
    """
    # spaCyのTokenなら.text、そうでなければそのまま使う
    word = word.text if hasattr(word, "text") else str(word)
    lemma = lemmatizer.lemmatize(word, pos="v")

    irregular_verbs = {
        "go": {"past": "went", "3sg": "goes", "ing": "going"},
        "eat": {"past": "ate", "3sg": "eats", "ing": "eating"},
        "run": {"past": "ran", "3sg": "runs", "ing": "running"},
        "swim": {"past": "swam", "3sg": "swims", "ing": "swimming"},
        "be": {"past": "was", "3sg": "is", "ing": "being"},
        "have": {"past": "had", "3sg": "has", "ing": "having"},
    }

    for base, forms in irregular_verbs.items():
        for form_type, form in forms.items():
            if word == form:
                return form_type

    if word.endswith("ies") and lemma + "y" == word[:-3] + "y":
        return "3sg"
    if word.endswith(("s", "es")) and lemma + "s" == word:
        return "3sg"

    if word.endswith("ied") and lemma + "y" == word[:-3] + "y":
        return "past"
    if word.endswith("ed") and lemma + "ed" == word:
        return "past"

    if word.endswith("ing") and lemma + "ing" == word:
        return "ing"

    return "base"

# 動詞の型を修正する
def fallback_inflect(lemma, transformation):
    if transformation == "past":
        # 語尾が "e" の場合は "d" を追加（例: like -> liked）
        if lemma.endswith("e"):
            return lemma + "d"
        else:
            return lemma + "ed"
    elif transformation == "ing":
        # "ie" -> "ying" 例: lie -> lying
        if lemma.endswith("ie"):
            return lemma[:-2] + "ying"
        # "e"で終わるが "ee" は除外（例: make -> making, but see -> seeing）
        elif lemma.endswith("e") and not lemma.endswith("ee"):
            return lemma[:-1] + "ing"
        else:
            return lemma + "ing"
    elif transformation == "3sg":
        # 語尾が "y" で前が子音の場合は "ies"（例: try -> tries）
        if lemma.endswith("y") and len(lemma) > 1 and lemma[-2] not in "aeiou":
            return lemma[:-1] + "ies"
        # s, sh, ch, x, z で終わる場合は "es"（例: watch -> watches）
        elif lemma.endswith(("s", "sh", "ch", "x", "z")):
            return lemma + "es"
        else:
            return lemma + "s"
    else:
        return lemma
    
def change_verb_form_nltk(sentence, transformation):
    """
    文中の動詞を指定した形に活用変化させる関数。

    :param sentence: 変化させたい英文 (例: "She run to school")
    :param transformation: 変化の種類 ("3sg" = 三単現, "past" = 過去形, "ing" = 現在分詞)
    :return: 変化後の文
    """
    doc = nlp(sentence)
    new_words = []

    for token in doc:
        if token.pos_ == "VERB":
            # SpaCyのlemmaで原形を取得
            lemma = token.lemma_
            # word_formsを使用して変換候補を取得
            wf = get_word_forms(lemma)

            if transformation == "3sg":
                forms = list(wf.get("third_person_singular", []))
            elif transformation == "past":
                forms = list(wf.get("past", []))
            elif transformation == "ing":
                forms = list(wf.get("present_participle", []))
            else:
                forms = []

            if forms:
                new_word = forms[0]
            else:
                new_word = fallback_inflect(lemma, transformation)

            new_words.append(new_word)
        else:
            new_words.append(token.text)

    return " ".join(new_words)

# 文章の違和感をスコアにする
"""
def calculate_perplexity(text):
    # モデルを評価モードに
    model.eval()

    # トークン化（パディング無し、短めの最大長）
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,  # 高速化のため短くする
        padding=False,
    )

    # GPUに転送（もし使っていれば）
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),  # 無くてもOKなモデルもある
            labels=inputs["input_ids"],
        )
        loss = outputs.loss
        perplexity = torch.exp(loss)

    return perplexity.item()
"""
### それぞれのページについて


@app.route("/")
def home():
    return render_template("home.html")

@app.route("/practice", methods=["GET", "POST"])
def practice():
    level = request.args.get("level")
    # CSVファイルを読み込む(単語リスト)
    df_words = pd.read_csv(CSV_MAP[level], encoding="utf-8")

    # ランダムに書籍を選択
    number_book = random.randint(0, 24)
    text = df.iloc[number_book, 2]  # 2列目に英文がある
    # 正規表現を使って文単位に分割（".", "?", "!" で区切る）
    sentences = re.split(r"(?<=[.!?])\s+", text)
    # ランダムに文章の番号を選択
    number_sentence = random.randint(0, len(sentences) - 1)
    first_sentence = sentences[number_sentence]
    ### ここまでで、prompt元の文章を取ってくることができた

    ### ここからは、文章を生成するパート
    generated_text = ""
    generated_text = generate_text(first_sentence, max_new_tokens=100, temperature=0.7)
    ### ここまでで、文章を生成できた

    ### ここからは、文章をチェックするパート
    # 文章の品詞判定
    doc = nlp(generated_text)

    HATENA_Block = ""  # 空欄とする単語
    HATENA_POS = ""  # 空欄とする単語の品詞
    verb_type = ""
    musikui_text = generated_text  # 空欄の入った文章

    # df_words の単語と一致する単語が文章内にあるか確認
    matching_words = []
    for i in range(len(df_words)):
        word = df_words.iloc[i, 0]  # 該当するレベルのリストの単語
        expected_pos = df_words.iloc[i, 1]  # 該当するレベルのリストの品詞

        for token in doc:  # 文章内の各単語を調べる
            if (
                token.lemma_.lower()
                == word.lower()  # 単語の一致を確認（大文字小文字を区別しない）
                and token.pos_ == expected_pos  # 品詞の一致を確認
            ):
                matching_words.append((token, word, expected_pos))

    # 一致する単語があれば、ランダムに1つ選ぶ
    if matching_words:
        # ランダムに選択
        select_number = random.randint(0, len(matching_words) - 1)
        HATENA_Block = matching_words[select_number][0]
        HATENA_POS = matching_words[select_number][2]

        # もし、HATENA_POSが動詞("VERB")であれば、動詞の型を判定する
        if HATENA_POS == "VERB":
            verb_type = detect_verb_form(HATENA_Block)

        # 置換 : 完全一致した単語のみ(   )に置換
        musikui_text = re.sub(
            rf"\b{re.escape(str(HATENA_Block))}\b",
            "(   )",
            generated_text,
            flags=re.IGNORECASE,
        )
        TrueorFalse = True
    else:
        TrueorFalse = False

    # もし問題文が生成されなかったら、
    if TrueorFalse == False:
        musikui_text = "Sorry. Please go back to the home and try again."

    ### ここからはダミーの選択肢を作る

    # ダミー1(同じ品詞のものから取る)
    dummy1 = dummy_samepos(HATENA_Block, HATENA_POS, level)
    # もし動詞の場合は、動詞の型を正解のものと揃える
    if HATENA_POS == "VERB":
        dummy1 = change_verb_form_nltk(dummy1, verb_type)

    # ダミー2(同じ品詞のものから取る)
    dummy2 = dummy_samepos(HATENA_Block, HATENA_POS, level)
    # もし動詞の場合は、動詞の型を正解のものと揃える
    if HATENA_POS == "VERB":
        dummy2 = change_verb_form_nltk(dummy2, verb_type)

    # ダミー3(品詞に関係なく単語を取ってくる)
    dummy3 = dummy_normal(HATENA_Block, level)
    # もし動詞の場合は、動詞の型を正解のものと揃える
    if HATENA_POS == "VERB":
        dummy3 = change_verb_form_nltk(dummy3, verb_type)
    # 4/24　HATENA_Blockをstr型に変更してjson形式に変化できるようにした
    List = [
        [str(HATENA_Block), "正解"],
        [dummy1, "不正解"],
        [dummy2, "不正解"],
        [dummy3, "不正解"],
    ]

    # シャッフルする
    random.shuffle(List)


    return render_template(
        "practice.html",
        generated_text=generated_text,
        musikui_text=musikui_text,
        level=level,
        correct=HATENA_Block,
        options=List
    )

@app.route("/submit", methods=["GET", "POST"])
def submit():
    level = request.form.get("level")
    generated_text = request.form.get("generated_text")
    musikui_text = request.form.get("musikui_text")
    user_answer = request.form.get("answer")
    correct = request.form.get("correct")


    # 元の単語を戻して表示したい
    highlighted_text = generated_text.replace(
        correct, f'<span style="color: #02A08F; font-weight: bold;">{correct}</span>'
    )
    # HTMLとしてレンダリングできるようにマークアップ
    generated_text_with_highlight = Markup(highlighted_text)

    return render_template(
        "result.html",
        level=level,
        generated_text=generated_text,
        musikui_text=musikui_text,
        generated_text_with_highlight=generated_text_with_highlight,
        user_answer=user_answer,
        correct=correct,
    )

# if __name__ == "__main__":
#     app.run(debug=True)

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000)

if __name__ == "__main__":
    # 本番モードで起動（デバッグ・リロード無効化）
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)