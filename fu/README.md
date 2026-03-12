# ./root/fu

# 🀄 Python Riichi Mahjong Simulation

A comprehensive object-oriented simulation of Japanese Riichi Mahjong built in Python. This project aims to accurately replicate standard JMA/EMA rulesets, including precise state management, hand evaluation (Shanten/Ukeire), and scoring logic.

## 📋 Features

- **Standard Ruleset**: Adheres to modern Japanese Riichi rules (Ton-Nan war).
- **Complete Game Loop**: Handles dealing, drawing, discarding, calling (Naki), and settling.
- **Advanced Mechanics**: Implements Furiten, Riichi, Dora, Kan, and Honba logic.
- **Hand Evaluation**: Algorithms for Shanten calculation and Wait analysis.
- **Scoring Engine**: Accurate Han/Fu calculation and Yaku validation.

## 📜 Ruleset Specification

This simulation follows standard **4-Player Riichi Mahjong** rules.

### General
- **Tiles**: 136 tiles (No flower/season tiles).
- **Players**: 4 (East, South, West, North).
- **Rounds**: East Round (Tonba) → South Round (Nanba). Game ends after South 4 (All Last).
- **Starting Score**: 25,000 points per player.

### Deal & Dora
- **Dealer**: Receives 14 tiles; others receive 13.
- **Dead Wall**: 14 tiles reserved (7 pairs).
- **Dora Indicator**: 1 tile revealed at start. Additional indicators revealed upon every Kan.

### Calls (Naki)
- **Ron**: Priority based on turn order (Atama Hane). Furiten applies.
- **Pon**: Open meld of 3 identical tiles.
- **Kan**: 
  - **Ankan**: Closed quad (retains Menzen status).
  - **Daiminkan**: Open quad on discard.
- **Chi**: **Only allowed from the player to the immediate left**. Open sequence of 3.

### Special States
- **Riichi**: Declared on Menzen Tenpai. Costs 1,000 points.
- **Furiten**: 
  - **Permanent**: Cannot Ron if waiting on a tile previously discarded by self.
  - **Temporary**: Cannot Ron until next self-draw after discarding.
- **Honba**: 300-point counter sticks added per Renchan.

## 🏗 Architecture

The project is structured into five core components:

```text
┌─────────────────────────────────────────────────────────┐
│                     GameEngine                          │
│  (Manages State Machine: Init → Loop → Settlement)      │
└─────────────────────────────────────────────────────────┘
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
┌─────────────────┐ ┌─────────────┐ ┌─────────────────┐
│      Table      │ │    Player   │ │      Hand       │
│ - Wall          │ │ - Score     │ │ - Tiles         │
│ - Dead Wall     │ │ - Wind      │ │ - Melds         │
│ - Dora          │ │ - Flags     │ │ - Shanten       │
│ - Discards      │ │ - AI Logic  │ │ - Wait Analysis │
└─────────────────┘ └─────────────┘ └─────────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │     Tile    │
                    │ - Suit      │
                    │ - Rank      │
                    │ - Red Dora  │
                    └─────────────┘
```

## Game Theory
- [ ] Mahjong Game Theory


- Expected Value Calculations
  - _Shanten_: Number of moves to winning (reach 0 shanten)
  - _Uke-ire_: Number of tiles which can advance my _shanten_
  - _Yaku_: How many points that hand will score
  - _Safety_: How safe a card to discard is

# Library
[Qiita Mahjong Lib Examples](https://qiita.com/FJyusk56/items/8189bcca3849532d095f)
[Mahjong RL Environment](https://github.com/smly/RiichiEnv)

# How to run code
1. `./` root dir
2. `cd fu`
3. `uv sync` > `uv run [file_name.py]`

