import sys
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QPushButton,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
)
from PyQt5.QtCore import Qt
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from toy_hearthstone import setup_game, describe_action
from hs_core.dqn_play_agent import DQNPlayAgent
from hs_core.random_agent import RandomAgent
import copy


class GameGUI(QMainWindow):
    def __init__(self, agent1, agent2, manual_player=0):
        super().__init__()
        self.setWindowTitle("Hearthstone Toy Game GUI")
        self.agent1 = agent1
        self.agent2 = agent2
        self.manual_player = manual_player  # 0 or 1
        self.state = setup_game()
        self.state.players[0].start_turn()
        self.history = []
        self.last_opponent_actions = []  # To store what the opponent played last turn
        self.init_ui()
        self.update_board()

    def create_card_widget(
        self, name, attack=None, health=None, width=90, height=120, hidden=False
    ):
        from PyQt5.QtWidgets import QVBoxLayout, QFrame

        card = QFrame()
        card.setFrameShape(QFrame.Box)
        card.setLineWidth(2)
        card.setFixedSize(width, height)
        layout = QVBoxLayout(card)
        label = QLabel()
        if hidden:
            label.setText("?")
        else:
            label.setText(name)
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
        if attack is not None and health is not None and not hidden:
            stats = QLabel(f"{attack}/{health}")
            stats.setAlignment(Qt.AlignCenter)
            layout.addWidget(stats)
        return card

    def create_board_row(self, minions):
        from PyQt5.QtWidgets import QHBoxLayout, QWidget

        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setSpacing(10)
        layout.setContentsMargins(0, 0, 0, 0)
        for m in minions:
            layout.addWidget(self.create_card_widget(m.card.name, m.attack, m.health))
        return row

    def create_hand_row(self, cards, hidden=False):
        from PyQt5.QtWidgets import QHBoxLayout, QWidget

        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setSpacing(10)
        layout.setContentsMargins(0, 0, 0, 0)
        for c in cards:
            layout.addWidget(
                self.create_card_widget(c.name if not hidden else "?", hidden=hidden)
            )
        return row

    def create_deck_widget(self, count, label):
        from PyQt5.QtWidgets import QVBoxLayout, QFrame

        deck = QFrame()
        deck.setFrameShape(QFrame.Box)
        deck.setLineWidth(2)
        deck.setFixedSize(60, 90)
        layout = QVBoxLayout(deck)
        deck_label = QLabel(f"{label}\n{count}")
        deck_label.setAlignment(Qt.AlignCenter)
        font = deck_label.font()
        font.setPointSize(11)
        deck_label.setFont(font)
        layout.addWidget(deck_label)
        return deck

    def init_ui(self):
        from PyQt5.QtWidgets import (
            QGridLayout,
            QHBoxLayout,
            QVBoxLayout,
            QTextEdit,
            QSizePolicy,
        )

        central = QWidget()
        self.setCentralWidget(central)
        self.setFixedSize(1280, 800)
        main_grid = QGridLayout()
        main_grid.setContentsMargins(0, 0, 0, 0)
        main_grid.setSpacing(0)
        central.setLayout(main_grid)

        # Board background
        board_bg = QWidget()
        board_bg.setStyleSheet("background-color: #e0c080;")
        board_layout = QGridLayout()
        board_layout.setContentsMargins(40, 40, 40, 40)
        board_bg.setLayout(board_layout)
        main_grid.addWidget(board_bg, 0, 0, 10, 10)

        # Opponent hand (top)
        self.opp_hand_row = QWidget()
        opp_hand_layout = QHBoxLayout(self.opp_hand_row)
        opp_hand_layout.setSpacing(10)
        opp_hand_layout.setContentsMargins(0, 0, 0, 0)
        board_layout.addWidget(self.opp_hand_row, 0, 2, 1, 6, alignment=Qt.AlignHCenter)

        # Opponent hero, mana, deck (top center)
        opp_top_row = QHBoxLayout()
        self.opp_hero_label = QLabel()
        self.opp_hero_label.setAlignment(Qt.AlignCenter)
        self.opp_hero_label.setFixedSize(80, 80)
        self.opp_hero_label.setStyleSheet(
            "background: #fff2; border-radius: 40px; border: 2px solid #a00;"
        )
        opp_top_row.addWidget(self.opp_hero_label)
        self.opp_mana_label = QLabel()
        self.opp_mana_label.setAlignment(Qt.AlignCenter)
        opp_top_row.addWidget(self.opp_mana_label)
        self.opp_deck = self.create_deck_widget(0, "Deck")
        opp_top_row.addWidget(self.opp_deck)
        opp_top_row.addStretch(1)
        board_layout.addLayout(opp_top_row, 1, 7, 1, 2, alignment=Qt.AlignRight)

        # Opponent board (row 2)
        self.opp_board_row = QWidget()
        opp_board_layout = QHBoxLayout(self.opp_board_row)
        opp_board_layout.setSpacing(30)
        opp_board_layout.setContentsMargins(0, 0, 0, 0)
        board_layout.addWidget(
            self.opp_board_row, 2, 2, 1, 6, alignment=Qt.AlignHCenter
        )

        # End turn button (right, vertically centered)
        self.end_turn_btn = QPushButton("END TURN")
        self.end_turn_btn.setStyleSheet(
            "font-size: 22px; background: #bfff80; border: 3px solid #0c0; border-radius: 12px; padding: 16px;"
        )
        self.end_turn_btn.setFixedSize(180, 60)
        self.end_turn_btn.clicked.connect(self.end_turn)
        board_layout.addWidget(self.end_turn_btn, 4, 8, 2, 2, alignment=Qt.AlignVCenter)

        # Player board (row 3)
        self.player_board_row = QWidget()
        player_board_layout = QHBoxLayout(self.player_board_row)
        player_board_layout.setSpacing(30)
        player_board_layout.setContentsMargins(0, 0, 0, 0)
        board_layout.addWidget(
            self.player_board_row, 6, 2, 1, 6, alignment=Qt.AlignHCenter
        )

        # Player hero, mana, deck (bottom center)
        player_bot_row = QHBoxLayout()
        self.player_hero_label = QLabel()
        self.player_hero_label.setAlignment(Qt.AlignCenter)
        self.player_hero_label.setFixedSize(80, 80)
        self.player_hero_label.setStyleSheet(
            "background: #fff2; border-radius: 40px; border: 2px solid #00a;"
        )
        player_bot_row.addWidget(self.player_hero_label)
        self.player_mana_label = QLabel()
        self.player_mana_label.setAlignment(Qt.AlignCenter)
        player_bot_row.addWidget(self.player_mana_label)
        self.player_deck = self.create_deck_widget(0, "Deck")
        player_bot_row.addWidget(self.player_deck)
        player_bot_row.addStretch(1)
        board_layout.addLayout(player_bot_row, 8, 7, 1, 2, alignment=Qt.AlignRight)

        # Player hand (bottom)
        self.player_hand_row = QWidget()
        player_hand_layout = QHBoxLayout(self.player_hand_row)
        player_hand_layout.setSpacing(10)
        player_hand_layout.setContentsMargins(0, 0, 0, 0)
        board_layout.addWidget(
            self.player_hand_row, 9, 2, 1, 6, alignment=Qt.AlignHCenter
        )

        # Info label (top left)
        self.info_label = QLabel()
        self.info_label.setAlignment(Qt.AlignLeft)
        self.info_label.setMinimumHeight(30)
        board_layout.addWidget(self.info_label, 0, 0, 1, 2)

        # Opponent action label (top right)
        self.opponent_action_label = QLabel()
        self.opponent_action_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.opponent_action_label.setMinimumHeight(40)
        board_layout.addWidget(self.opponent_action_label, 0, 8, 1, 2)

        # Action list (bottom left)
        self.action_list = QListWidget()
        self.action_list.setMinimumHeight(60)
        self.action_list.setMaximumWidth(300)
        board_layout.addWidget(self.action_list, 9, 0, 1, 2)

        # Play action button (bottom left)
        self.play_btn = QPushButton("Play Selected Action")
        self.play_btn.clicked.connect(self.play_action)
        board_layout.addWidget(self.play_btn, 9, 8, 1, 2)

    def update_board(self):
        from PyQt5.QtWidgets import QHBoxLayout

        p1, p2 = self.state.players[0], self.state.players[1]
        info = f"Turn {p1.turn + p2.turn} - Player {self.state.current + 1}'s turn"
        self.info_label.setText(info)

        # Opponent hand (hidden or visible)
        for i in reversed(
            range(
                self.opp_hand_row.layout().count() if self.opp_hand_row.layout() else 0
            )
        ):
            self.opp_hand_row.layout().itemAt(i).widget().setParent(None)
        opp_hand_layout = QHBoxLayout()
        opp_hand_layout.setSpacing(10)
        opp_hand_layout.setContentsMargins(0, 0, 0, 0)
        if self.manual_player == 0:
            for _ in p2.hand:
                opp_hand_layout.addWidget(self.create_card_widget("?", hidden=True))
        else:
            for c in p1.hand:
                opp_hand_layout.addWidget(self.create_card_widget(c.name))
        self.opp_hand_row.setLayout(opp_hand_layout)

        # Opponent deck
        self.opp_deck.layout().itemAt(0).widget().setText(f"Deck\n{len(p2.deck)}")

        # Opponent hero HP
        self.opp_hero_label.setText(f"HP: {p2.hero_hp}")

        # Opponent mana
        self.opp_mana_label.setText(f"Mana: {p2.mana}/{p2.max_mana}")

        # Opponent board
        for i in reversed(
            range(
                self.opp_board_row.layout().count()
                if self.opp_board_row.layout()
                else 0
            )
        ):
            self.opp_board_row.layout().itemAt(i).widget().setParent(None)
        opp_board_layout = QHBoxLayout()
        opp_board_layout.setSpacing(10)
        opp_board_layout.setContentsMargins(0, 0, 0, 0)
        for m in p2.board:
            opp_board_layout.addWidget(
                self.create_card_widget(m.card.name, m.attack, m.health)
            )
        # Ensure board row stretches to fill available space
        opp_board_layout.addStretch(1)
        self.opp_board_row.setLayout(opp_board_layout)

        # Player board
        for i in reversed(
            range(
                self.player_board_row.layout().count()
                if self.player_board_row.layout()
                else 0
            )
        ):
            self.player_board_row.layout().itemAt(i).widget().setParent(None)
        player_board_layout = QHBoxLayout()
        player_board_layout.setSpacing(10)
        player_board_layout.setContentsMargins(0, 0, 0, 0)
        for m in p1.board:
            player_board_layout.addWidget(
                self.create_card_widget(m.card.name, m.attack, m.health)
            )
        # Ensure board row stretches to fill available space
        player_board_layout.addStretch(1)
        self.player_board_row.setLayout(player_board_layout)

        # Player hero HP
        self.player_hero_label.setText(f"HP: {p1.hero_hp}")

        # Player mana
        self.player_mana_label.setText(f"Mana: {p1.mana}/{p1.max_mana}")

        # Player hand (visible)
        for i in reversed(
            range(
                self.player_hand_row.layout().count()
                if self.player_hand_row.layout()
                else 0
            )
        ):
            self.player_hand_row.layout().itemAt(i).widget().setParent(None)
        player_hand_layout = QHBoxLayout()
        player_hand_layout.setSpacing(10)
        player_hand_layout.setContentsMargins(0, 0, 0, 0)
        for c in p1.hand:
            player_hand_layout.addWidget(self.create_card_widget(c.name))
        self.player_hand_row.setLayout(player_hand_layout)

        # Player deck
        self.player_deck.layout().itemAt(0).widget().setText(f"Deck\n{len(p1.deck)}")

        # Log area (show last 10 actions)
        log_str = ""
        for prev_state, action in self.history[-10:]:
            log_str += f"P{1 if prev_state.current == 0 else 2}: {describe_action(prev_state, action, prev_state, prev_state.current)}\n"
        if hasattr(self, "log_area"):
            self.log_area.setPlainText(log_str)

        # Opponent action label
        if self.last_opponent_actions:
            self.opponent_action_label.setText(
                "Opponent's last turn:\n" + "\n".join(self.last_opponent_actions)
            )
        else:
            self.opponent_action_label.setText("")

        self.populate_actions()

    def populate_actions(self):
        self.action_list.clear()
        actions = self.state.get_legal_actions()
        for i, action in enumerate(actions):
            desc = describe_action(self.state, action, self.state, self.state.current)
            item = QListWidgetItem(f"{i}: {desc}")
            item.setData(Qt.UserRole, action)
            self.action_list.addItem(item)

    def play_action(self):
        selected = self.action_list.currentItem()
        if not selected:
            QMessageBox.warning(self, "No Action", "Please select an action.")
            return
        action = selected.data(Qt.UserRole)
        # prev_state = copy.deepcopy(self.state)  # For consistency/history, but not used directly
        prev_state = copy.deepcopy(self.state)
        self.state.step(action)
        self.history.append((prev_state, action))
        if self.state.is_terminal():
            self.update_board()
            QMessageBox.information(
                self, "Game Over", f"Winner: Player {self.state.winner + 1}"
            )
            self.close()
            return
        if action[0] == "end":
            self.next_turn()
        else:
            self.update_board()

    def end_turn(self):
        # Save prev_state before stepping, so the log shows the correct player
        prev_state = copy.deepcopy(self.state)
        self.state.step(("end",))
        self.history.append((prev_state, ("end",)))
        if self.state.is_terminal():
            self.update_board()
            QMessageBox.information(
                self, "Game Over", f"Winner: Player {self.state.winner + 1}"
            )
            self.close()
            return
        self.next_turn()

    def next_turn(self):
        # If it's the manual player's turn, do nothing (wait for input)
        if self.state.current == self.manual_player:
            self.update_board()
            return
        # Otherwise, let the agent play automatically
        agent = self.agent1 if self.state.current == 0 else self.agent2
        self.last_opponent_actions = []
        prev_state = copy.deepcopy(self.state)
        actions_taken = []
        while True:
            actions = self.state.get_legal_actions()
            non_end_actions = [a for a in actions if a[0] != "end"]
            if not non_end_actions:
                action = ("end",)
            else:
                action = agent.choose_action(self.state)
            prev_state_turn = copy.deepcopy(self.state)
            self.state.step(action)
            self.history.append((prev_state_turn, action))
            # Only record actions that are not 'end'
            if action[0] != "end":
                desc = describe_action(
                    self.state, action, prev_state_turn, prev_state_turn.current
                )
                actions_taken.append(desc)
            if self.state.is_terminal():
                self.last_opponent_actions = actions_taken
                self.update_board()
                QMessageBox.information(
                    self, "Game Over", f"Winner: Player {self.state.winner + 1}"
                )
                self.close()
                return
            if action[0] == "end":
                break
        self.last_opponent_actions = actions_taken
        self.update_board()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    dqn_agent = DQNPlayAgent(
        state_dim=54, action_dim=50, model_path="dqn_policy/best.pt"
    )
    random_agent = RandomAgent()
    window = GameGUI(agent1=random_agent, agent2=dqn_agent, manual_player=0)
    window.show()
    sys.exit(app.exec_())
