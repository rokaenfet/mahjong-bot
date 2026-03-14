"""GameState dataclass capturing all recognized information from a single screenshot."""

from __future__ import annotations

from dataclasses import dataclass, field

from tiles import Tile


@dataclass
class GameState:
    # --- Own hand ---
    hand: list[Tile | None] = field(default_factory=list)
    drawn_tile: Tile | None = None
    self_melds: list[list[Tile]] = field(default_factory=list)   # pon/chii/kan groups

    # --- Discards ---
    self_discards: list[Tile] = field(default_factory=list)
    # [right_player, across_player, left_player]
    opponent_discards: list[list[Tile]] = field(default_factory=lambda: [[], [], []])

    # --- Opponent melds ---
    # [right_player, across_player, left_player][group_index][tile]
    opponent_melds: list[list[list[Tile]]] = field(default_factory=lambda: [[], [], []])

    # --- Center-table info ---
    dora_indicators: list[Tile] = field(default_factory=list)
    tiles_remaining: int | None = None
    round_wind: str | None = None    # "east" / "south" / "west" / "north"
    round_number: int | None = None  # 1-4 within the round wind

    # --- Per-player info ---
    # Keys: "self", "right", "across", "left"
    seat_winds: dict[str, str] = field(default_factory=dict)
    scores: dict[str, int | None] = field(default_factory=dict)
    riichi_status: dict[str, bool] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = []

        # Round & wind
        if self.round_wind and self.round_number:
            lines.append(f"Round: {self.round_wind.capitalize()} {self.round_number}")

        # Scores & winds
        for pos in ("self", "right", "across", "left"):
            wind = self.seat_winds.get(pos, "?")
            score = self.scores.get(pos)
            riichi = " [RIICHI]" if self.riichi_status.get(pos) else ""
            score_str = str(score) if score is not None else "?"
            lines.append(f"  {pos:6s}: {wind:5s}  {score_str:>6}{riichi}")

        # Dora
        if self.dora_indicators:
            lines.append(f"Dora indicators: {[str(t) for t in self.dora_indicators]}")

        # Tiles remaining
        if self.tiles_remaining is not None:
            lines.append(f"Tiles remaining: {self.tiles_remaining}")

        # Own hand
        hand_str = ", ".join(str(t) if t is not None else "???" for t in self.hand)
        drawn_str = f"  +  Drawn: {self.drawn_tile}" if self.drawn_tile else ""
        lines.append(f"Hand: [{hand_str}]{drawn_str}")
        if self.self_melds:
            melds_str = " | ".join(
                "[" + ", ".join(str(t) for t in g) + "]" for g in self.self_melds
            )
            lines.append(f"Self melds: {melds_str}")

        # Discards
        if self.self_discards:
            lines.append(f"Self discards: {[str(t) for t in self.self_discards]}")
        for i, pos in enumerate(("right", "across", "left")):
            discards = self.opponent_discards[i]
            if discards:
                lines.append(f"{pos.capitalize()} discards: {[str(t) for t in discards]}")
        for i, pos in enumerate(("right", "across", "left")):
            melds = self.opponent_melds[i]
            if melds:
                melds_str = " | ".join(
                    "[" + ", ".join(str(t) for t in g) + "]" for g in melds
                )
                lines.append(f"{pos.capitalize()} melds: {melds_str}")

        return "\n".join(lines)
