# ./root/fu

# Functionalities
- [ ] Mahjong Game Theory
   - [ ] Dataset of all hands
   - [ ] Dataset of all scoring patterns

# Notes
- Mahjong Words
  - 当たり牌（あたりはい）
  - 筒子（ぴんず）
  - 萬子（わんず）
  - 索子（そーず）
  - 嶺上（りんしゃん）
  - 海底（はいてい）
  - 河底（ほうてい）
  - 場風牌（ばふうはい）
  - 自風牌（じふうはい）
  - ドラ
  - 裏ドラ
  - 赤ドラ
  - 槓ドラ
  - 自摸（ツモ）
  - 河（かわ）
  - 打牌（だはい）
  - 捨牌（すてはい）
  - ツモ切り（つもぎり）
  - ポン
  - チー
  - 明カン（みんかん）
  - 暗カン（あんかん）
  - 頭（あたま）
  - 面子（メンツ）
  - 塔子（ターツ）
  - 対子（トイツ）
  - 順子（シュンツ）
  - 刻子（コーツ）
  - 槓子（カンツ）
  - 暗刻（アンコ）
  - 明刻（ミンコ）
  - 暗槓（アンカン）
  - 明槓（ミンカン）
  - 和り（あがり）
  - ロン
  - 門前清自摸和（メンゼンツモ）
  - 一発（イッパツ）
  - 手役（テヤク）
  - 放銃（ホウジュウ）
  - 嵌張待ち（カンチャンマチ）
  - 辺張待ち（ペンチャンマチ）
  - 両面待ち（リャンメンマチ）
  - 単騎待ち（タンキマチ）
  - 双碰待ち（シャンポンマチ）
  - ノベタン
  - 裸単騎（ハダカタンキ）
  - 聴牌（テンパイ）
  - 二向聴（リャンシャンテン）
  - 不聴（ノーテン）
  - 立直（リーチ）
  - ダブルリーチ
  - 黙聴（ダマテン）
  - 振聴（フリテン）
  - 筋（スジ）
  - 
  - 翻（ハン）
  - 符（フ）
  - 牌（パイ）
  - 安全牌（あんぜんぱい）
  - 危険牌（きけんはい）
  - 上がり牌（あがりはい）
  - 三元牌（さんげんぱい）
  - 風牌（かぜはい）
  - 字牌（じはい）
  - 数牌（しゅうぱい）
  - 副底（フーテイ）
  - 満貫（まんがん）
  - 跳満（はねまん）
  - 倍満（ばいまん）
  - 三倍満（さんばいまん）
  - 役満（やくまん）
  - 立直（リーチ）
  - 門前（メンゼン）
  - 食い下がり
  - 喰いタン

- For my turn, decide...
  - Actions
    - Discard
    - Naki (An-Kan)
    - Ri-Chi
- For opponent turn which I can interact with (Naki), decide...
  - Actions
    - Naki (Pon, Chii, Kan)
    - Ron

- Expected Value Calculations
  - _Shanten_: Number of moves to winning (reach 0 shanten)
  - _Uke-ire_: Number of tiles which can advance my _shanten_
  - _Yaku_: How many points that hand will score
  - _Safety_: How safe a card to discard is

**Maximizing Uke-ire maximizes Shante, higher chance of finishing** 

# Library
[Qiita Mahjong Lib Examples](https://qiita.com/FJyusk56/items/8189bcca3849532d095f)
[Mahjong RL Environment](https://github.com/smly/RiichiEnv)

# How to run code
1. `./` root dir
2. `cd fu`
3. `uv sync` > `uv run [file_name.py]`

# Folder Struct
```
Project Tree: fu
├── mahjong_soul_tiles.png
├── main.py
├── pyproject.toml        
├── README.md
├── test.ipynb
├── tile_imgs
│   ├── fuuhai
│   │   ├── w1.png        
│   │   ├── w2.png        
│   │   ├── w3.png        
│   │   └── w4.png        
│   ├── manzu
│   │   ├── m1.png        
│   │   ├── m2.png        
│   │   ├── m3.png        
│   │   ├── m4.png        
│   │   ├── m5.png        
│   │   ├── m6.png        
│   │   ├── m7.png        
│   │   ├── m8.png        
│   │   └── m9.png        
│   ├── pinzu
│   │   ├── p1.png        
│   │   ├── p2.png
│   │   ├── p3.png
│   │   ├── p4.png
│   │   ├── p5.png
│   │   ├── p6.png
│   │   ├── p7.png
│   │   ├── p8.png
│   │   └── p9.png
│   ├── sanngennhai
│   │   ├── d1.png
│   │   ├── d2.png
│   │   └── d3.png
│   └── sozu
│       ├── s1.png
│       ├── s2.png
│       ├── s3.png
│       ├── s4.png
│       ├── s5.png
│       ├── s6.png
│       ├── s7.png
│       └── s8.png
├── type_imgs
│   ├── f.JPG
│   ├── m.JPG
│   ├── p.JPG
│   ├── s.JPG
│   └── sn.JPG
└── uv.lock
```