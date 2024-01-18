"""
Simple example pokerbot, written in Python.
"""
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction, BidAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot
import random
import eval7


class Player(Bot):
    """
    A pokerbot.
    """

    def __init__(self):
        """
        Called when a new game starts. Called exactly once.

        Arguments:
        Nothing.

        Returns:
        Nothing.
        """
        # debugging
        self.folds = 0
        self.preflops = 0

        # opp stats
        self.opp_folds = 0
        self.opp_preflops = 0


    def handle_new_round(self, game_state, round_state, active):
        """
        Called when a new round starts. Called NUM_ROUNDS times.

        Arguments:
        game_state: the GameState object.
        round_state: the RoundState object.
        active: your player's index.

        Returns:
        Nothing.
        """
        # my_bankroll = game_state.bankroll  # the total number of chips you've gained or lost from the beginning of the game to the start of this round
        # game_clock = game_state.game_clock  # the total number of seconds your bot has left to play this game
        # round_num = game_state.round_num  # the round number from 1 to NUM_ROUNDS
        # my_cards = round_state.hands[active]  # your cards
        self.big_blind = bool(active)  # True if you are the big blind
        print(f'---round {game_state.round_num}---')
        self.folded = False
        self.opp_preflop_opportunity = True


    def handle_round_over(self, game_state, terminal_state, active):
        """
        Called when a round ends. Called NUM_ROUNDS times.

        Arguments:
        game_state: the GameState object.
        terminal_state: the TerminalState object.
        active: your player's index.

        Returns:
        Nothing.
        """
        # my_delta = terminal_state.deltas[active]  # your bankroll change from this round
        previous_state = terminal_state.previous_state  # RoundState before payoffs
        # street = previous_state.street  # 0, 3, 4, or 5 representing when this round ended
        # my_cards = previous_state.hands[active]  # your cards
        # opp_cards = previous_state.hands[1-active]  # opponent's cards or [] if not revealed
        pass
        if game_state.round_num == 1000:
            print(f'proportion preflop folds: {self.folds/self.preflops}')
            print(f'opponents fold rate: {self.opp_folds/self.opp_preflops}')
        if self.opp_preflop_opportunity:
            self.opp_preflops += 1
            if previous_state.street==0 and previous_state.button in [0,1] and not self.folded:
                self.opp_folds += 1

    
    def preflop_estimate(self, hand, iters):
        """
        hand (list): two cards
        iters (int): number of monte carlo iterations
        """
        deck = eval7.Deck()
        my_cards = [eval7.Card(card) for card in hand]
        for card in my_cards:
            deck.cards.remove(card)
        wins = 0

        for i in range(iters):
            deck.shuffle()
            board_cards = deck[:5]
            opp_cards = deck[5:7]
            if eval7.evaluate(my_cards+board_cards) > eval7.evaluate(opp_cards+board_cards):
                wins += 1

        return wins/iters
    
    def auction_estimate(self, hand, flop, iters):
        """
        hand (list): your cards (length 2)
        flop (list): cards on the board (length 3)
        iters (int): number of monte carlo iterations
        """
        deck = eval7.Deck()
        my_cards = [eval7.Card(card) for card in hand]
        flop_cards = [eval7.Card(card) for card in flop]
        for card in my_cards+flop_cards:
            deck.cards.remove(card)
        wins3,wins2 = 0,0

        for i in range(iters):
            deck.shuffle()
            unflipped_cards = deck[0:2]
            auction = deck[2]
            opp_cards = deck[3:5]
            val2 = eval7.evaluate(my_cards+flop_cards+unflipped_cards)
            val3 = eval7.evaluate(my_cards+flop_cards+unflipped_cards+[auction])
            opp2 = eval7.evaluate(opp_cards+flop_cards+unflipped_cards)
            opp3 = eval7.evaluate(opp_cards+flop_cards+unflipped_cards+[auction])
            if val3 > opp2: # we win auction
                wins3 += 1
            if val2 > opp3: # we lose auction
                wins2 += 1
        return wins3/iters, wins2/iters


    def round_estimate(self, hand, opp, board, iters):
        """
        hand (list): your cards (length 2 or 3)
        opp (int): number of cards your opponnet has (2 or 3)
        board (list): cards on the board (length 3, 4, or 5)
        iters (int): number of monte carlo iterations
        """
        deck = eval7.Deck()
        my_cards = [eval7.Card(card) for card in hand]
        board_cards = [eval7.Card(card) for card in board]
        for card in my_cards+board_cards:
            deck.cards.remove(card)
        wins,losses,ties = 0,0,0

        for i in range(iters):
            deck.shuffle()
            unflipped = 5-len(board)
            board_cards += deck[:unflipped]
            opp_cards = deck[unflipped:unflipped+2] if opp==2 else deck[unflipped:unflipped+3]
            my_val = eval7.evaluate(my_cards+board_cards)
            opp_val = eval7.evaluate(opp_cards+board_cards)
            if my_val > opp_val:
                wins +=1
            if my_val < opp_val:
                losses += 1
            else:
                ties += 1

        return wins/iters,losses/iters
    

    def get_action(self, game_state, round_state, active):
        """
        Where the magic happens - your code should implement this function.
        Called any time the engine needs an action from your bot.

        Arguments:
        game_state: the GameState object.
        round_state: the RoundState object.
        active: your player's index.

        Returns:
        Your action.
        """
        # May be useful, but you may choose to not use.
        legal_actions = round_state.legal_actions() # the actions you are allowed to take
        street = round_state.street # 0, 3, 4, or 5 representing pre-flop, flop, turn, or river respectively
        my_cards = round_state.hands[active]  # your cards
        board_cards = round_state.deck[:street]  # the board cards
        my_pip = round_state.pips[active]  # the number of chips you have contributed to the pot this round of betting
        opp_pip = round_state.pips[1 - active]  # the number of chips your opponent has contributed to the pot this round of betting
        my_stack = round_state.stacks[active]  # the number of chips you have remaining
        opp_stack = round_state.stacks[1 - active]  # the number of chips your opponent has remaining
        my_bid = round_state.bids[active]  # How much you bid previously (available only after auction)
        opp_bid = round_state.bids[1 - active]  # How much opponent bid previously (available only after auction)
        continue_cost = (opp_pip - my_pip)  # the number of chips needed to stay in the pot
        my_contribution = STARTING_STACK - my_stack  # the number of chips you have contributed to the pot
        opp_contribution = STARTING_STACK - opp_stack  # the number of chips your opponent has contributed to the pot
        effective_stack = min(my_stack, opp_stack)

        if RaiseAction in legal_actions:
            min_raise, max_raise = round_state.raise_bounds() # the smallest and largest numbers of chips for a legal bet/raise
            min_cost = min_raise - my_pip  # the cost of a minimum bet/raise
            max_cost = max_raise - my_pip  # the cost of a maximum bet/raise
        
        # preflop
        if street == 0:
            if round_state.button in [0,1]:
                self.preflops += 1
                self.p_win = self.preflop_estimate(my_cards, 100)
                self.cutoff = 0.575
                # if game_state.round_num > 300 and self.opp_folds/self.opp_preflops>0.9:
                #     cutoff = min(0.7, self.opp_folds/self.opp_preflops + 0.1)
                #     print(f'{cutoff=}')
                if self.p_win < self.cutoff and FoldAction in legal_actions:
                    if round_state.button==0:
                        self.opp_preflop_opportunity = False
                    self.folds += 1
                    self.folded = True
                    return FoldAction()
            if self.p_win >= self.cutoff and RaiseAction in legal_actions and round_state.button<2:
                # print(min_raise,min_raise+int((self.p_win-0.5)*(max_raise-min_raise)),max_raise)
                return RaiseAction(random.randint(min_raise,min_raise+int((self.p_win-0.5)*(max_raise-min_raise))))
            elif CheckAction in legal_actions:
                return CheckAction()
            else:
                return CallAction()

        # auction or right after
        elif BidAction in legal_actions:
            if self.p_win < self.cutoff:
                return BidAction(0)
            return BidAction(my_stack)
            wins3,wins2 = self.auction_estimate(my_cards, board_cards, 100)
            auction_val = int(750*(wins3-wins2))
            print(f'{auction_val=}')
            if my_stack > opp_stack:
                print(f'{opp_stack=}')
                # auction_val = min(auction_val, opp_stack+1)
                auction_val = opp_stack+1
            else:
                print(f'{my_stack=}')
                auction_val = min(auction_val, my_stack)
            auction_val = max(auction_val,0)
            print(f'{auction_val=}')
            return BidAction(auction_val)

        # normal round
        opp = 2 if my_bid > opp_bid else 3
        p_win,p_lose = self.round_estimate(my_cards,opp,board_cards,100)
        if p_win*opp_contribution - p_lose*(my_contribution+continue_cost) < -1*my_contribution:
            return FoldAction()
        if RaiseAction in legal_actions and random.random()<0.3:
            raise_amt = int((p_win-p_lose)*effective_stack)
            raise_amt = min(raise_amt,max_raise)
            if raise_amt < min_raise+1:
                return RaiseAction(min_raise)
            else:
                return RaiseAction(random.randint(min_raise,raise_amt))
        if CheckAction in legal_actions:
            return CheckAction()
        return CallAction()


if __name__ == "__main__":
    run_bot(Player(), parse_args())
