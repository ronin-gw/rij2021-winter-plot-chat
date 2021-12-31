#!/usr/bin/env python3
import sys
import json
import re
import os.path
import pickle
from datetime import datetime, timezone, timedelta
from collections import Counter
from itertools import chain
from multiprocessing import Pool
from operator import itemgetter
from copy import copy

from sudachipy import tokenizer, dictionary
import jaconv

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator
from matplotlib.font_manager import FontProperties

from adjustText import adjust_text

from emoji import UNICODE_EMOJI

matplotlib.use("module://mplcairo.macosx")

TIMELINE = os.path.join(os.path.dirname(__file__), "timeline.pickle")
TIMEZONE = timezone(timedelta(hours=9), "JST")

matplotlib.rcParams["font.sans-serif"] = ["Hiragino Maru Gothic Pro", "Yu Gothic", "Meirio", "Takao", "IPAexGothic", "IPAPGothic", "VL PGothic", "Noto Sans CJK JP"]
emoji_prop = FontProperties(fname="/System/Library/Fonts/Apple Color Emoji.ttc")

UNICODE_EMOJI = UNICODE_EMOJI["en"]

# (ward to plot, line style, color)
RTA_EMOTES = (
    ("rtaClap", "-", "#ec7087"),
    ("rtaPray", "-", "#f7f97a"),
    ("rtaHatena", "-", "#ffb5a1"),
    ("rtaGl", "-", "#5cc200"),
    ("rtaGg", "-", "#ff381c"),
    ("rtaCheer", "-", "#ffbe00"),
    ("rtaR", "-", "white"),
    (("rtaCry", "BibleThump"), "-", "#5ec6ff"),

    ("rtaFear", "-.", "#8aa0ec"),
    ("rtaListen", "-.", "#5eb0ff"),
    ("rtaPokan", "-.", "#838187"),
    ("rtaKabe", "-.", "#bf927a"),
    ("rtaMaru", "-.", "#c80730"),
    ("rtaPog", "-.", "#f8c900"),
    (("rtaRedbull", "rtaRedbull2"), "-.", "#98b0df"),
    ("rtaPolice", "-.", "#7891b8"),
    ("rtaGogo", ":", "#df4f69"),
    ("rtaFrameperfect", ":", "#ff7401"),
    ("rtaPixelperfect", ":", "#ffa300"),
    ("rtaBanana", ":", "#f3f905"),
    ("rtaBatsu", ":", "#5aafdd"),
    ("rtaShogi", ":", "#c68d46"),
    ("rtaIizo", ":", "#0f9619"),

    # ("rtaHello", "-.", "#ff3291"),
    # ("rtaHmm", "-.", "#fcc7b9"),
    # ("rtaOko", "-.", "#d20025"),
    # ("rtaWut", "-.", "#d97f8d"),
    # ("rtaChan", "-.", "green"),
    # ("rtaKappa", "-.", "#ffeae2"),

    # ("rtaSleep", "-.", "#ff8000"),
    # ("rtaCafe", "--", "#a44242"),
    # ("rtaDot", "--", "#ff3291"),

    # ("rtaShi", ":", "#8aa0ec"),
    # ("rtaGift", ":", "white"),
    # ("rtaAnkimo", ":", "#f92218"),

    (("草", "ｗｗｗ", "LUL"), "--", "green"),
    ("無敵時間", "--", "red"),
    ("かわいい", "--", "#ff3291"),
    ("ゴルフ", "--", "#e2b444"),
    ("ファイナル", "--", "gray"),
    (("PokSuicune", "スイクン"), "--", "#c38cdc"),
    ("Squid4", "--", "#80d2b4"),
    ("石油王", "--", "yellow"),
    (("ｆｆｆ", "稲"), "--", "#ffeab4"),
    ("〜ケンカ", "--", "orange"),
    # (("Kappu", "カップ", "日清食品"), "--", "#f9bc71"),
    # ("サクラチル", "--", "#ffe0e0"),

)
VOCABULARY = set(w for w, _, _, in RTA_EMOTES if isinstance(w, str))
VOCABULARY |= set(chain(*(w for w, _, _, in RTA_EMOTES if isinstance(w, tuple))))

# (title, movie start time as timestamp, offset hour, min, sec)
GAMES = (
    # ("始まりのあいさつ", 1628617360.7, 0, 12, 51),
    ("クロノトリガー", 1640453484.7, 0, 27, 51),
    ("天穂のサクナヒメ", 1640453484.7, 3, 35, 8),
    ("聴音RPG【失われた音問村】", 1640453484.7, 5, 27, 26),
    ("Racing\nPitch", 1640453484.7, 6, 30, 23),
    ("Trials Rising", 1640453484.7, 6, 49, 57),
    ("Don't\nSpill\nYour Coffee", 1640453484.7, 7, 17, 49),
    ("Minoria", 1640453484.7, 7, 29, 44),
    ("メイドさんを\n右にミ☆", 1640453484.7, 8, 9, 3),
    ("東方妖々夢\n～ Perfect\nCherry Blossom.", 1640453484.7, 8, 40, 23),
    ("Touhou Luna Nights", 1640453484.7, 9, 8, 12),
    ("WHAT THE GOLF?", 1640453484.7, 9, 57, 20),
    ("Google Doodle\nChampion Island\nGames Begin!", 1640453484.7, 11, 2, 28, "right"),
    ("Bloodstained:\nRitual of the Night", 1640453484.7, 11, 24, 26),
    ("Celeste", 1640453484.7, 12, 2, 59),
    ("Rabi-Ribi", 1640453484.7, 13, 17, 8),
    ("ドラゴンクエストⅤ", 1640453484.7, 15, 3, 46),
    ("ドラゴンクエスト", 1640453484.7, 22, 9, 28),
    ("ショックトルーパーズ\nセカンドスカッド", 1640453484.7, 23, 58, 36),
    ("ジャイロセット", 1640453484.7, 24, 54, 17),
    ("攻殻機動隊\nGHOS\nIN TH\nSHELL", 1640453484.7, 26, 6, 46),
    ("アーマード・コア３ サイレントライン", 1640453484.7, 26, 38, 33),
    ("Portal", 1640453484.7, 28, 3, 59),
    ("星のカービィ\n参上! ドロッチェ団", 1640453484.7, 28, 50, 5),
    ("星のカービィ\n夢の泉デラックス", 1640453484.7, 29, 37, 20),
    ("ワリオの森", 1640453484.7, 30, 13, 1),
    ("すーぱーぐっすんおよよ2", 1640453484.7, 30, 31, 0),
    ("NINTENDOパズルコレクション\n／パネルでポン", 1640453484.7, 31, 21, 37),
    ("ヨッシーのクッキー", 1640453484.7, 32, 29, 19),
    ("TETRIS", 1640453484.7, 33, 15, 58),
    ("Newポケモンスナップ", 1640453484.7, 34, 36, 40),
    ("ポケットモンスター クリスタル", 1640453484.7, 37, 36, 21),
    ("Final Fantasy Ⅲ (3D Remake)", 1640601776.7, 0, 7, 13),
    ("イース・オリジン", 1640601776.7, 4, 15, 29),
    ("ブレイヴフェンサー 武蔵伝", 1640601776.7, 5, 54, 46),
    ("ファイアーエムブレム 風花雪月", 1640601776.7, 8, 15, 55),
    ("スーパーペーパーマリオ", 1640601776.7, 9, 59, 7, "right"),
    ("スーパーマリオ 3Dワールド\n + フューリーワールド", 1640601776.7, 14, 8, 45),
    ("スーパーマリオ６４", 1640601776.7, 15, 23, 29),
    ("マイクタイソン・\nパンチアウト！！", 1640601776.7, 17, 36, 56),
    ("マイティボンジャック\n/ Mighty Bomb Jack", 1640601776.7, 18, 17, 7),
    ("BIOHAZARD Operation Raccoon City", 1640601776.7, 18, 56, 45),
    ("BIOHAZARD VILLAGE", 1640601776.7, 20, 14, 10),
    ("Mad Rat Dead", 1640601776.7, 22, 7, 30),
    ("Devil May Cry 5: Special Edition", 1640601776.7, 24, 18, 11),
    ("ダークソウル3", 1640601776.7, 25, 19, 46),
    ("Metal Gear Solid", 1640601776.7, 26, 56, 53),
    ("ロックマンエグゼ６", 1640601776.7, 27, 40, 47),
    ("Rosenkreuzstilette", 1640601776.7, 30, 10, 59),
    ("Dance Dance Revolution EXTRA MIX", 1640601776.7, 30, 54, 5),
    ("Sekiro: Shadows Die Twice", 1640601776.7, 32, 36, 43),
    ("ソニック・ザ・\nヘッジホッグ 2", 1640601776.7, 33, 55, 50),
    ("海腹川背 Fresh!", 1640601776.7, 34, 28, 24),
    ("ドンキーコング64", 1640601776.7, 35, 11, 35),
    ("塊魂アンコール", 1640601776.7, 37, 28, 14),
    ("不思議のダンジョン 風来のシレン5plus\nフォーチュンタワーと運命のダイス", 1640601776.7, 38, 10, 57),
    ("ドラゴンクエスト３", 1640601776.7, 39, 14, 44),
    ("ファイナルファンタジーVI", 1640601776.7, 39, 48, 1),
    ("R4\n-RIDGE RACER TYPE4-", 1640601776.7, 40, 57, 58),
    ("F-ZERO X", 1640601776.7, 41, 37, 30),
    ("ゼルダの伝説\nふしぎの木の実\n大地の章&時空の章", 1640601776.7, 42, 34, 3, "right"),
    ("ゼルダの伝説ムジュラの仮面", 1640601776.7, 44, 18, 14),
    ("すってはっくん\n（ロムカセット版）", 1640767281.7, 0, 3, 17),
    ("テイルズオブグレイセスｆ", 1640767281.7, 0, 48, 57),
    ("オクトパストラベラー", 1640767281.7, 4, 57, 8),
    ("信長の野望・革新 with パワーアップキット", 1640767281.7, 9, 2, 1, "right"),
    ("実況パワフルプロ野球6", 1640767281.7, 10, 51, 20),
    ("モンスターファームアドバンス", 1640767281.7, 11, 51, 30),
    ("Rez Infinite", 1640767281.7, 13, 30, 28),
    ("メタルスラッグ初代", 1640767281.7, 14, 36, 27, "right"),
    ("パリ・ダカール・\nラリー・スペシャル", 1640767281.7, 15, 12, 48, "right"),
    ("Nippon Marathon", 1640767281.7, 15, 50, 46, "right"),
    ("キテレツ大百科", 1640767281.7, 16, 50, 49),
    ("美少女戦士\nセーラームーンR", 1640767281.7, 17, 19, 5),
    ("スプラトゥーン", 1640767281.7, 17, 54, 32),
    ("スーパーマリオワールド", 1640767281.7, 19, 0, 13),
    ("月風魔伝", 1640767281.7, 20, 35, 18),
    ("ホーリー\nダイヴァー", 1640767281.7, 21, 15, 51),
    ("迦楼羅王", 1640767281.7, 21, 42, 48),
    ("NiGHTS\ninto dreams...HD", 1640767281.7, 22, 21, 39, "right"),
    ("Cook, Serve, Delicious! 3?!", 1640767281.7, 22, 53, 8),
    ("花咲か妖精フリージア", 1640767281.7, 24, 56, 30),
    ("電車でＤ ClimaxStage", 1640767281.7, 25, 46, 27),
    ("バトルトード", 1640767281.7, 26, 26, 14),
    ("Spelunker", 1640767281.7, 27, 26, 24),
    ("魔界村駅伝リレー", 1640767281.7, 28, 1, 59),
    ("イースセルセタの樹海", 1640767281.7, 30, 21, 41),
    ("ドルアーガの塔", 1640767281.7, 31, 26, 11),
    ("ソロモンの鍵", 1640767281.7, 32, 18, 55),
    ("グロブダー", 1640767281.7, 33, 13, 39),
    ("怒首領蜂最大往生", 1640767281.7, 34, 34, 31),
    ("星のカービィ スーパーデラックス", 1640767281.7, 35, 54, 4),
    ("スーパードンキーコング3", 1640767281.7, 37, 15, 22),
    ("ゼルダの伝説\n時のオカリナ", 1640767281.7, 39, 20, 39),
    ("ファイナルソード DefinitiveEdition", 1640767281.7, 39, 50, 19),
    ("終わりのあいさつ", 1640767281.7, 41, 49, 11, "right")
)


class Game:
    def __init__(self, name, t, h, m, s, align="left"):
        self.name = name
        self.startat = datetime.fromtimestamp(t + h * 3600 + m * 60 + s)
        self.align = align


GAMES = tuple(Game(*args) for args in GAMES)

WINDOWSIZE = 1
WINDOW = timedelta(seconds=WINDOWSIZE)
AVR_WINDOW = 60
PER_SECONDS = 60
FIND_WINDOW = 15
DOMINATION_RATE = 0.6
COUNT_THRESHOLD = 45

DPI = 200
ROW = 5
PAGES = 4
YMAX = 700
WIDTH = 3840
HEIGHT = 2160

FONT_COLOR = "white"
FRAME_COLOR = "#ffff79"
BACKGROUND_COLOR = "#352319"
FACE_COLOR = "#482b1e"
ARROW_COLOR = "#ffff79"
MESSAGE_FILL_COLOR = "#1e0d0b"


class Message:
    _tokenizer = dictionary.Dictionary().create()
    _mode = tokenizer.Tokenizer.SplitMode.C

    pns = (
        "無敵時間",
        "石油王",
        "かわいい",
        "納期のテーマ",
        "あんなもの",
        "スナップしろ",
        "見損なったぞカーネル",
        "被害者の会",
        "なっとるやろがい",
        "待機時間",
        "不死身時間",
        "三銃士",
        "キテレツ"
    )
    pn_patterns = (
        (re.compile("[\u30A1-\u30FF]+ケンカ"), "〜ケンカ"),
    )
    stop_words = (
        "Squid2",
    )

    @classmethod
    def _tokenize(cls, text):
        return cls._tokenizer.tokenize(text, cls._mode)

    def __init__(self, raw):
        self.name = raw["author"]["name"]
        if "emotes" in raw:
            self.emotes = set(e["name"] for e in raw["emotes"]
                              if e["name"] not in self.stop_words)
        else:
            self.emotes = set()
        self.datetime = datetime.fromtimestamp(int(raw["timestamp"]) // 1000000).replace(tzinfo=TIMEZONE)

        self.message = raw["message"]
        self.msg = set()

        message = self.message
        for emote in self.emotes:
            message = message.replace(emote, "")
        for stop in self.stop_words:
            message = message.replace(stop, "")

        #
        for pattern, replace in self.pn_patterns:
            match = pattern.findall(message)
            if match:
                self.msg.add(replace)
                for m in match:
                    message.replace(m, "")

        #
        for pn in self.pns:
            if pn in message:
                self.msg.add(pn)
                message = message.replace(pn, "")

        #
        message = jaconv.h2z(message)

        # (名詞 or 動詞) (+助動詞)を取り出す
        parts = []
        currentpart = None
        for m in self._tokenize(message):
            part = m.part_of_speech()[0]

            if currentpart:
                if part == "助動詞":
                    parts.append(m.surface())
                else:
                    self.msg.add(''.join(parts))
                    parts = []
                    if part in ("名詞", "動詞"):
                        currentpart = part
                        parts.append(m.surface())
                    else:
                        currentpart = None
            else:
                if part in ("名詞", "動詞"):
                    currentpart = part
                    parts.append(m.surface())

        if parts:
            self.msg.add(''.join(parts))

        #
        kusa = False
        for word in copy(self.msg):
            if set(word) & set(('w', 'ｗ')):
                kusa = True
                self.msg.remove(word)
        if kusa:
            self.msg.add("ｗｗｗ")

        #
        ine = False
        for word in copy(self.msg):
            if set(word) & set(('f', 'ｆ')):
                ine = True
                self.msg.remove(word)
        if ine:
            self.msg.add("ｆｆｆ")

        message = message.strip()
        if not self.msg and message:
            self.msg.add(message)

    def __len__(self):
        return len(self.msg)

    @property
    def words(self):
        return self.msg | self.emotes


def _parse_chat(paths):
    messages = []
    for p in paths:
        with open(p) as f, Pool() as pool:
            j = json.load(f)
            messages += list(pool.map(Message, j, len(j) // pool._processes))

    timeline = []
    currentwindow = messages[0].datetime.replace(microsecond=0) + WINDOW
    _messages = []
    for m in messages:
        if m.datetime <= currentwindow:
            _messages.append(m)
        else:
            timeline.append((currentwindow, *_make_timepoint(_messages)))
            while True:
                currentwindow += WINDOW
                if m.datetime <= currentwindow:
                    _messages = [m]
                    break
                else:
                    timeline.append((currentwindow, 0, Counter()))

    if _messages:
        timeline.append((currentwindow, *_make_timepoint(_messages)))

    return timeline


def _make_timepoint(messages):
    total = len(messages)
    counts = Counter(_ for _ in chain(*(m.words for m in messages)))

    return total, counts


def _load_timeline(paths):
    if os.path.exists(TIMELINE):
        with open(TIMELINE, "rb") as f:
            timeline = pickle.load(f)
    else:
        timeline = _parse_chat(paths)
        with open(TIMELINE, "wb") as f:
            pickle.dump(timeline, f)

    return timeline


def _save_counts(timeline):
    _, _, counters = zip(*timeline)

    counter = Counter()
    for c in counters:
        counter.update(c)

    with open("words.tab", 'w') as f:
        for w, c in sorted(counter.items(), key=itemgetter(1), reverse=True):
            print(w, c, sep='\t', file=f)


def _plot(timeline):
    for npage in range(1, 1 + PAGES):
        chunklen = int(len(timeline) / PAGES / ROW)

        fig = plt.figure(figsize=(WIDTH / DPI, HEIGHT / DPI), dpi=DPI)
        fig.patch.set_facecolor(BACKGROUND_COLOR)
        plt.rcParams["savefig.facecolor"] = BACKGROUND_COLOR
        plt.subplots_adjust(left=0.07, bottom=0.05, top=0.92)

        for i in range(1, 1 + ROW):
            nrow = i + ROW * (npage - 1)
            f, t = chunklen * (nrow - 1), chunklen * nrow
            x, c, y = zip(*timeline[f:t])
            _x = tuple(t.replace(tzinfo=None) for t in x)

            ax = fig.add_subplot(ROW, 1, i)
            _plot_row(ax, _x, y, c, i == 1, i == ROW)

        fig.suptitle(f"RTA in Japan Winter 2021 チャット頻出スタンプ・単語 ({npage}/{PAGES})",
                     color=FONT_COLOR, size="x-large")
        fig.text(0.03, 0.5, "単語 / 分 （同一メッセージ内の重複は除外）",
                 ha="center", va="center", rotation="vertical", color=FONT_COLOR, size="large")
        fig.savefig(f"{npage}.png", dpi=DPI)
        plt.close()
        print(npage)


def moving_average(x, w=AVR_WINDOW):
    _x = np.convolve(x, np.ones(w), "same") / w
    return _x[:len(x)]


def _plot_row(ax, x, y, total_raw, add_upper_legend, add_lower_legend):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M", tz=TIMEZONE))
    ax.xaxis.set_major_locator(mdates.HourLocator())
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(range(0, 60, 5)))
    ax.yaxis.set_minor_locator(MultipleLocator(50))
    ax.set_facecolor(FACE_COLOR)
    for axis in ("top", "bottom", "left", "right"):
        ax.spines[axis].set_color(FRAME_COLOR)

    ax.tick_params(colors=FONT_COLOR, which="both")
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0, YMAX)

    total = moving_average(total_raw) * PER_SECONDS
    total = ax.fill_between(x, 0, total, color=BACKGROUND_COLOR)

    for i, game in enumerate(GAMES):
        if x[0] <= game.startat <= x[-1]:
            #
            if i == 10:
                # ax.axvline(x=game.startat, color=ARROW_COLOR, linestyle=":")
                yfrom = - YMAX / 14
                yto = YMAX * 0.85
                flag_x = mdates.date2num(game.startat)
                width = flag_x - mdates.date2num(game.startat - (x[-1] - x[0]) / 100)

                patch = matplotlib.patches.Ellipse(xy=(game.startat, yfrom), width=width, height=-yfrom / 2, color="black", fill=True, clip_on=False)
                line = matplotlib.lines.Line2D((game.startat, game.startat), (yfrom, yto), linewidth=2, color="white", clip_on=False, alpha=0.8)
                flag = matplotlib.patches.Polygon(xy=((flag_x, yto), (flag_x, yto - YMAX / 7), (flag_x - width * 2, yto - YMAX / 14)), closed=True, color="#e2b444")
                ax.add_patch(patch)
                ax.add_line(line)
                ax.add_patch(flag)
                ax.annotate(' ' + game.name, xy=(game.startat, YMAX * 0.85), verticalalignment="top",
                            color=FONT_COLOR, arrowprops=dict(width=0, headwidth=0), ha=game.align)
            else:
                ax.axvline(x=game.startat, color=ARROW_COLOR, linestyle=":")
                ax.annotate(game.name, xy=(game.startat, YMAX), xytext=(game.startat, YMAX * 0.85), verticalalignment="top",
                            color=FONT_COLOR, arrowprops=dict(facecolor=ARROW_COLOR, shrink=0.05), ha=game.align)

    # ys = []
    # labels = []
    # colors = []
    for words, style, color in RTA_EMOTES:
        if isinstance(words, str):
            words = (words, )
        _y = np.fromiter((sum(c[w] for w in words) for c in y), int)
        if not sum(_y):
            continue
        _y = moving_average(_y) * PER_SECONDS
        # ys.append(_y)
        # labels.append("\n".join(words))
        # colors.append(color if color else None)
        ax.plot(x, _y, label="\n".join(words), linestyle=style, color=(color if color else None))
    # ax.stackplot(x, ys, labels=labels, colors=colors)

    #
    avr_10min = moving_average(total_raw, FIND_WINDOW) * FIND_WINDOW
    words = Counter()
    for counter in y:
        words.update(counter)
    words = set(k for k, v in words.items() if v >= COUNT_THRESHOLD)
    words -= VOCABULARY

    annotations = []
    for word in words:
        at = []
        _ys = moving_average(np.fromiter((c[word] for c in y), int), FIND_WINDOW) * FIND_WINDOW
        for i, (_y, total_y) in enumerate(zip(_ys, avr_10min)):
            if _y >= total_y * DOMINATION_RATE and _y >= COUNT_THRESHOLD:
                at.append((i, _y * PER_SECONDS / FIND_WINDOW))
        if at:
            at.sort(key=lambda x: x[1])
            at = at[-1]

            if any(c in UNICODE_EMOJI for c in word):
                text = ax.text(x[at[0]], at[1], word, color=FONT_COLOR, fontsize="xx-small", fontproperties=emoji_prop)
            else:
                text = ax.text(x[at[0]], at[1], word, color=FONT_COLOR, fontsize="xx-small")
            annotations.append(text)
    adjust_text(annotations, only_move={"text": 'x'})

    if add_upper_legend:
        leg = ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
        _set_legend(leg)

    if add_lower_legend:
        leg = plt.legend([total], ["メッセージ / 分"], loc=(1.015, 0.5))
        _set_legend(leg)
        msg = "図中の単語は{}秒間で{}%の\nメッセージに含まれていた単語\n({}メッセージ / 秒 以上のもの)".format(
            FIND_WINDOW, int(DOMINATION_RATE * 100), int(COUNT_THRESHOLD / FIND_WINDOW)
        )
        plt.gcf().text(0.92, 0.08, msg, fontsize="x-small", color=FONT_COLOR)


def _set_legend(leg):
    frame = leg.get_frame()
    frame.set_facecolor(FACE_COLOR)
    frame.set_edgecolor(FRAME_COLOR)

    for text in leg.get_texts():
        text.set_color(FONT_COLOR)


def _main():
    timeline = _load_timeline(sys.argv[1:])
    _save_counts(timeline)
    _plot(timeline)


if __name__ == "__main__":
    _main()
