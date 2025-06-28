class AggroScriptedAgent:
    """Always attacks face if possible, otherwise plays highest-cost card, else ends turn."""

    def choose_action(self, state):
        actions = state.get_legal_actions()
        # Prioritize attacking face
        for a in actions:
            if a[0] == "attack" and a[2] == "hero":
                return a
        # Play highest-cost card
        play_actions = [a for a in actions if a[0] == "play"]
        if play_actions:
            hand = state.players[state.current].hand
            return max(play_actions, key=lambda x: hand[x[1]].cost)
        return ("end",)


class BoardControlScriptedAgent:
    """Always attacks minions if possible, otherwise plays highest-cost card, else ends turn."""

    def choose_action(self, state):
        actions = state.get_legal_actions()
        # Prioritize attacking minions
        for a in actions:
            if a[0] == "attack" and a[2] != "hero":
                return a
        # Play highest-cost card
        play_actions = [a for a in actions if a[0] == "play"]
        if play_actions:
            hand = state.players[state.current].hand
            return max(play_actions, key=lambda x: hand[x[1]].cost)
        # If can attack face, do so
        for a in actions:
            if a[0] == "attack" and a[2] == "hero":
                return a
        return ("end",)


class BalancedScriptedAgent:
    """Attacks minions if outnumbered, otherwise attacks face. Plays highest-cost card."""

    def choose_action(self, state):
        actions = state.get_legal_actions()
        player = state.players[state.current]
        opponent = state.players[1 - state.current]

        # 1. If outnumbered, clear board (attack minions)
        if len(opponent.board) > len(player.board):
            # Attack minions if possible
            for a in actions:
                if a[0] == "attack" and a[2] != "hero":
                    return a
            # Play minions that deal damage to opponent's minions if favorable
            play_minion_actions = [
                a
                for a in actions
                if a[0] == "play" and player.hand[a[1]].card_type == "minion"
            ]
            for a in play_minion_actions:
                card = player.hand[a[1]]
                # Check if minion has a deal_X effect and can target an opponent minion
                if (
                    hasattr(card, "effect")
                    and card.effect
                    and "deal" in card.effect
                    and a[2] is not None
                    and a[2] != "hero"
                ):
                    # Find the target minion
                    try:
                        target_idx = a[2]
                        target_minion = opponent.board[target_idx]
                        # Parse effect amount
                        effect = card.effect
                        if "deal_" in effect:
                            amount = int(effect.split("deal_")[1])
                            # If the effect can kill the minion, play it
                            if amount >= target_minion.health:
                                return a
                    except Exception:
                        pass
            # Otherwise, play highest-cost minion to develop board
            if play_minion_actions:
                hand = player.hand
                return max(play_minion_actions, key=lambda x: hand[x[1]].cost)

        # 2. If opponent hero HP is low, attack face
        if opponent.hero_hp <= 2:
            for a in actions:
                if a[0] == "attack" and a[2] == "hero":
                    return a

        # 3. If can clear a minion with a spell, do so
        play_spell_actions = [
            a
            for a in actions
            if a[0] == "play" and player.hand[a[1]].card_type == "spell"
        ]
        for a in play_spell_actions:
            # If the spell targets a minion, prefer it when outnumbered
            if a[2] != "hero":
                return a

        # 4. If can attack face, do so
        for a in actions:
            if a[0] == "attack" and a[2] == "hero":
                return a

        # 5. Play highest-cost minion to develop board
        play_minion_actions = [
            a
            for a in actions
            if a[0] == "play" and player.hand[a[1]].card_type == "minion"
        ]
        if play_minion_actions:
            hand = player.hand
            return max(play_minion_actions, key=lambda x: hand[x[1]].cost)

        # 6. Play highest-cost spell (if any)
        if play_spell_actions:
            hand = player.hand
            return max(play_spell_actions, key=lambda x: hand[x[1]].cost)

        # 7. Otherwise, attack minions if possible
        for a in actions:
            if a[0] == "attack" and a[2] != "hero":
                return a

        return ("end",)
