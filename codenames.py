import re
import time

import numpy as np
import gensim.downloader as api

from typing import List, Tuple


# ============================================================
# GLOVE
# ============================================================

print("Carregando modelo de word embeddings (GloVe)...")

try:
    VECTOR_MODEL = api.load("glove-wiki-gigaword-50")
    print("Modelo carregado com sucesso!")
except Exception as e:
    print(f"Erro ao carregar modelo vetorial: {e}")
    VECTOR_MODEL = None


# ============================================================
# CODEMASTER
# ============================================================

class SeuCodemaster:

    SEARCH_TIME = 60.0

    def __init__(self, team="Red", **kwargs):
        self.team = team

        self.model = VECTOR_MODEL

        self.board_words = []
        self.team_words = []
        self.enemy_words = []
        self.neutral_words = []
        self.assassin_word = ""

        self.move_history = None

    # --------------------------------------------------------
    # Framework
    # --------------------------------------------------------

    def set_move_history(self, move_history):
        self.move_history = move_history

    def set_game_state(
        self,
        words_on_board: List[str],
        key_grid: List[str]
    ) -> None:

        self.board_words = [
            word.lower().strip()
            for word in words_on_board
        ]

        self.team_words = []
        self.enemy_words = []
        self.neutral_words = []
        self.assassin_word = ""

        for word, key in zip(self.board_words, key_grid):

            key = str(key).upper().strip()

            if key == "ASSASSIN":
                self.assassin_word = word

            elif key in {"NEUTRAL", "CIVILIAN"}:
                self.neutral_words.append(word)

            elif key in {"RED", "BLUE"}:

                if key == self.team.upper():
                    self.team_words.append(word)
                else:
                    self.enemy_words.append(word)

            elif "TEAM" in key:
                self.team_words.append(word)

            elif "OPPONENT" in key or "ENEMY" in key:
                self.enemy_words.append(word)

        print(
            f"[Codemaster {self.team}] "
            f"TEAM: {self.team_words}"
        )

        print(
            f"[Codemaster {self.team}] "
            f"ENEMY: {self.enemy_words}"
        )

        print(
            f"[Codemaster {self.team}] "
            f"NEUTRAL: {self.neutral_words}"
        )

        print(
            f"[Codemaster {self.team}] "
            f"ASSASSIN: {self.assassin_word}"
        )

    # --------------------------------------------------------
    # Legal clue
    # --------------------------------------------------------

    def _is_legal_clue(
        self,
        clue: str
    ) -> bool:

        clue = clue.lower().strip()

        # Somente uma palavra formada por letras.
        if not re.fullmatch(r"^[a-z]+$", clue):
            return False

        # Não pode ser uma palavra visível no tabuleiro
        # nem uma derivação muito próxima dela.
        for board_word in self.board_words:

            board_word = board_word.lower().strip()

            if clue == board_word:
                return False

            if clue in board_word:
                return False

            if board_word in clue:
                return False

        return True

    # --------------------------------------------------------
    # Similaridade
    # --------------------------------------------------------

    def _similarity(
        self,
        clue: str,
        word: str
    ) -> float:

        if not self.model:
            return -1.0

        if clue not in self.model.key_to_index:
            return -1.0

        if word not in self.model.key_to_index:
            return -1.0

        try:
            return float(
                self.model.similarity(clue, word)
            )

        except Exception:
            return -1.0

    # --------------------------------------------------------
    # Busca vetorial
    # --------------------------------------------------------

    def _search_for_group(
        self,
        group: List[str],
        time_limit: float
    ):
        """
        Procura uma palavra que seja simultaneamente próxima
        de todas as palavras presentes em `group`.

        O score da candidata é a MENOR similaridade dela
        com qualquer palavra do grupo.

        Portanto:

            score = min(sim(clue, word))

        Isso força a candidata a possuir uma associação
        razoavelmente forte com todas as palavras do grupo,
        em vez de ser extremamente boa para apenas uma delas.

        A candidata também precisa respeitar:

            nenhuma palavra inimiga/neutra/assassino
            pode ser mais próxima do clue do que a pior
            palavra do nosso grupo.

        A busca percorre o vocabulário inteiro do GloVe.
        """

        if not self.model:
            return None

        valid_group = [
            word
            for word in group
            if word in self.model.key_to_index
        ]

        if not valid_group:
            return None

        # ----------------------------------------------------
        # Palavras proibidas
        # ----------------------------------------------------

        forbidden = (
            self.enemy_words
            + self.neutral_words
        )

        if self.assassin_word:
            forbidden.append(self.assassin_word)

        forbidden = [
            word
            for word in forbidden
            if word in self.model.key_to_index
        ]

        # ----------------------------------------------------
        # Matriz de vetores
        # ----------------------------------------------------

        group_vectors = np.array([
            self.model[word]
            for word in valid_group
        ])

        forbidden_vectors = np.array([
            self.model[word]
            for word in forbidden
        ])

        # Normalização
        group_norms = np.linalg.norm(
            group_vectors,
            axis=1,
            keepdims=True
        )

        group_vectors = (
            group_vectors / np.maximum(group_norms, 1e-12)
        )

        if len(forbidden_vectors) > 0:

            forbidden_norms = np.linalg.norm(
                forbidden_vectors,
                axis=1,
                keepdims=True
            )

            forbidden_vectors = (
                forbidden_vectors
                / np.maximum(forbidden_norms, 1e-12)
            )

        # ----------------------------------------------------
        # Vocabulário
        # ----------------------------------------------------

        vocabulary = self.model.index_to_key

        best_clue = None
        best_score = -np.inf

        start_time = time.time()

        # ----------------------------------------------------
        # Busca
        # ----------------------------------------------------

        for clue in vocabulary:

            # O objetivo é procurar durante o tempo permitido.
            if time.time() - start_time >= time_limit:
                break

            clue = clue.lower()

            # Não pode ser palavra do próprio tabuleiro.
            if not self._is_legal_clue(clue):
                continue

            clue_vector = self.model[clue]

            norm = np.linalg.norm(clue_vector)

            if norm <= 1e-12:
                continue

            clue_vector = clue_vector / norm

            # ------------------------------------------------
            # Similaridade com nosso grupo
            # ------------------------------------------------

            team_similarities = (
                group_vectors @ clue_vector
            )

            worst_team_similarity = float(
                np.min(team_similarities)
            )

            # ------------------------------------------------
            # Similaridade com inimigos/civis/assassino
            # ------------------------------------------------

            if len(forbidden_vectors) > 0:

                forbidden_similarities = (
                    forbidden_vectors @ clue_vector
                )

                best_forbidden_similarity = float(
                    np.max(forbidden_similarities)
                )

            else:
                best_forbidden_similarity = -1.0

            # ------------------------------------------------
            # Regra principal
            # ------------------------------------------------
            #
            # A pior palavra do nosso grupo precisa ser
            # mais próxima da dica do que QUALQUER palavra
            # proibida.
            #
            # Exemplo:
            #
            # team:
            #   0.72
            #   0.65
            #   0.61
            #
            # forbidden:
            #   0.58
            #
            # -> válido
            #
            # Mas:
            #
            # team:
            #   0.72
            #   0.65
            #   0.61
            #
            # forbidden:
            #   0.63
            #
            # -> inválido
            #

            if (
                best_forbidden_similarity
                >= worst_team_similarity
            ):
                continue

            # ------------------------------------------------
            # Melhor candidata encontrada
            # ------------------------------------------------

            if worst_team_similarity > best_score:

                best_score = worst_team_similarity
                best_clue = clue

        if best_clue is None:
            return None

        return (
            best_clue,
            best_score
        )

    # --------------------------------------------------------
    # Escolha da dica
    # --------------------------------------------------------

    def get_clue(self) -> Tuple[str, int]:

        if not self.model:
            return ("pass", 0)

        if not self.team_words:
            return ("pass", 0)

        print(
            f"[Codemaster {self.team}] "
            f"Iniciando busca vetorial..."
        )

        print(
            f"[Codemaster {self.team}] "
            f"Grupo inicial: {self.team_words}"
        )

        # ====================================================
        # PRIMEIRA FASE
        # ====================================================
        #
        # Procuramos SOMENTE uma dica que consiga relacionar
        # todas as palavras do nosso time.
        #
        # Não reduzimos o grupo durante esta fase.
        #
        # A busca pode durar até 60 segundos.
        # ====================================================

        result = self._search_for_group(
            self.team_words,
            self.SEARCH_TIME
        )

        if result is not None:

            clue, score = result

            print(
                f"[Codemaster {self.team}] "
                f"Dica encontrada para "
                f"{len(self.team_words)} palavras: "
                f"{clue} ({score:.4f})"
            )

            return (
                clue,
                len(self.team_words)
            )

        # ====================================================
        # SEGUNDA FASE
        # ====================================================
        #
        # Não conseguimos encontrar uma palavra que seja
        # simultaneamente segura para todas.
        #
        # Agora começamos a reduzir o grupo.
        #
        # 9 -> 8 -> 7 -> 6 -> ...
        # ====================================================

        print(
            f"[Codemaster {self.team}] "
            f"Nenhuma dica segura encontrada para "
            f"todas as palavras."
        )

        for group_size in range(
            len(self.team_words) - 1,
            0,
            -1
        ):

            print(
                f"[Codemaster {self.team}] "
                f"Tentando grupo de "
                f"{group_size} palavras..."
            )

            best_result = None
            best_group = None

            # ------------------------------------------------
            # Testamos todas as combinações possíveis.
            # ------------------------------------------------

            import itertools

            for group in itertools.combinations(
                self.team_words,
                group_size
            ):

                result = self._search_for_group(
                    list(group),
                    2.0
                )

                if result is None:
                    continue

                clue, score = result

                if (
                    best_result is None
                    or score > best_result[1]
                ):

                    best_result = (
                        clue,
                        score
                    )

                    best_group = group

            # ------------------------------------------------
            # Encontramos uma dica.
            # ------------------------------------------------

            if best_result is not None:

                clue, score = best_result

                print(
                    f"[Codemaster {self.team}] "
                    f"Dica encontrada: {clue}"
                )

                print(
                    f"[Codemaster {self.team}] "
                    f"Grupo: {best_group}"
                )

                print(
                    f"[Codemaster {self.team}] "
                    f"Score: {score:.4f}"
                )

                return (
                    clue,
                    group_size
                )

        # ====================================================
        # FALLBACK
        # ====================================================

        print(
            f"[Codemaster {self.team}] "
            f"Nenhuma dica segura encontrada."
        )

        return ("pass", 0)


# ============================================================
# GUESSER
# ============================================================

class SeuGuesser:

    def __init__(self, team="Red", **kwargs):

        self.team = team

        self.model = VECTOR_MODEL

        self.board_words = []

        self.current_clue = ""
        self.num_guesses = 0
        self.guesses_made = 0

        self.guess_candidates = []

        self.move_history = None

    # --------------------------------------------------------
    # Framework
    # --------------------------------------------------------

    def set_move_history(self, move_history):
        self.move_history = move_history

    def set_board(
        self,
        words: List[str]
    ) -> None:

        self.board_words = [
            word.lower().strip()
            for word in words
        ]

    # --------------------------------------------------------
    # Similaridade
    # --------------------------------------------------------

    def _similarity(
        self,
        clue: str,
        word: str
    ) -> float:

        if not self.model:
            return -1.0

        if clue not in self.model.key_to_index:
            return -1.0

        if word not in self.model.key_to_index:
            return -1.0

        try:
            return float(
                self.model.similarity(
                    clue,
                    word
                )
            )

        except Exception:
            return -1.0

    # --------------------------------------------------------
    # Nova dica
    # --------------------------------------------------------

    def set_clue(
        self,
        clue: str,
        num_guesses: int
    ) -> None:

        self.current_clue = clue.lower().strip()

        # No Codenames normalmente é permitido fazer
        # uma tentativa adicional.
        self.num_guesses = (
            num_guesses + 1
            if num_guesses > 0
            else 9
        )

        self.guesses_made = 0

        self.guess_candidates = []

        if not self.board_words:
            return

        if (
            not self.model
            or self.current_clue
            not in self.model.key_to_index
        ):

            self.guess_candidates = (
                list(self.board_words)
            )

            return

        # ----------------------------------------------------
        # Busca vetorial EXATA em todas as palavras do tabuleiro
        # ----------------------------------------------------

        scored = []

        for word in self.board_words:

            score = self._similarity(
                self.current_clue,
                word
            )

            scored.append(
                (word, score)
            )

        # Maior similaridade primeiro.
        scored.sort(
            key=lambda x: x[1],
            reverse=True
        )

        self.guess_candidates = [
            word
            for word, _ in scored
        ]

        print(
            f"[Guesser {self.team}] "
            f"Dica: {self.current_clue}"
        )

        print(
            f"[Guesser {self.team}] "
            f"Ranking vetorial:"
        )

        for word, score in scored:
            print(
                f"    {word}: {score:.4f}"
            )

    # --------------------------------------------------------
    # Continuar tentando?
    # --------------------------------------------------------

    def keep_guessing(self) -> bool:

        valid = [
            word
            for word in self.guess_candidates
            if word in self.board_words
        ]

        return (
            self.guesses_made
            < self.num_guesses
            and bool(valid)
        )

    # --------------------------------------------------------
    # Resposta
    # --------------------------------------------------------

    def get_answer(self) -> str:

        self.guesses_made += 1

        valid = [
            word
            for word in self.guess_candidates
            if word in self.board_words
        ]

        if valid:

            chosen = valid[0]

            self.guess_candidates.remove(
                chosen
            )

            return chosen

        if self.board_words:
            return self.board_words[0]

        return ""