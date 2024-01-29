"""
team jmoney_14  #swag
"""
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction, BidAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot
import random
import eval7
import torch
from torch import nn
from torch import optim

class Player(Bot):
    """
    A pokerbot.
    """

    def __init__(self):
        """
        Called when a new game starts. Called exactly once.

        Returns None
        """
        # my stats
        self.folds = 0
        self.preflops = 0
        self.cutoff = 0.575

        # opp stats
        self.opp_folds = 0
        self.opp_preflops = 0
        self.opp_bids = []
        self.my_pwins2 = []
        self.my_pwins3 = []
        self.preflop_raises = []
        self.opp_raises = []
        self.auction_model = nn.Sequential(
            nn.Linear(4,8),
            nn.ReLU(),
            nn.Linear(8,1),
            nn.ReLU()
        )


    def handle_new_round(self, game_state, round_state, active):
        """
        Called when a new round starts. Called NUM_ROUNDS times.

        game_state (obj): the GameState object.
        round_state (obj): the RoundState object.
        active (int): my player's index

        Returns None
        """
        # my_bankroll = game_state.bankroll  # the total number of chips you've gained or lost from the beginning of the game to the start of this round
        # game_clock = game_state.game_clock  # the total number of seconds your bot has left to play this game
        # round_num = game_state.round_num  # the round number from 1 to NUM_ROUNDS
        # my_cards = round_state.hands[active]  # your cards
        # self.big_blind = bool(active)  # True if you are the big blind
        print(f'---round {game_state.round_num}---')
        self.folded = False
        self.opp_preflop_opportunity = True


    def handle_round_over(self, game_state, terminal_state, active):
        """
        Called when a round ends. Called NUM_ROUNDS times.

        game_state (obj): the GameState object.
        terminal_state (obj): the TerminalState object.
        active (int): my player's index

        Returns None
        """
        # my_delta = terminal_state.deltas[active]  # your bankroll change from this round
        previous_state = terminal_state.previous_state  # RoundState before payoffs
        # street = previous_state.street  # 0, 3, 4, or 5 representing when this round ended
        # my_cards = previous_state.hands[active]  # your cards
        # opp_cards = previous_state.hands[1-active]  # opponent's cards or [] if not revealed
        
        # preflop folding stats
        if game_state.round_num == NUM_ROUNDS:
            print(f'proportion preflop folds: {self.folds/self.preflops}')
            print(f'opponents fold rate: {self.opp_folds/self.opp_preflops}')
        if self.opp_preflop_opportunity:
            self.opp_preflops += 1
            if previous_state.street==0 and previous_state.button in [0,1] and not self.folded:
                self.opp_folds += 1

        # auction model
        if terminal_state.bids != [None,None]:
            self.my_pwins2.append(self.p_win2)
            self.my_pwins3.append(self.p_win3)
            self.opp_bids.append(terminal_state.bids[1-active])
            total_raise,opp_raise = self.get_preflop_raises(previous_state,active)
            self.preflop_raises.append(total_raise)
            self.opp_raises.append(opp_raise)
        if game_state.round_num == 0.75*NUM_ROUNDS:
            print(f'training auction model, {self.auction_model}')
            self.train_auction_model()

    
    def preflop_estimate(self, hand, iters):
        """
        hand (list): two cards
        iters (int): number of monte carlo iterations

        Returns probability of winning given info so far
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
    

    def get_preflop_raises(self,game_state,active):
        """
        game_state (obj): game_state object
        active (int): my player's index

        Returns (total amount raised preflop, amount raised by opponent)
        """
        curr = game_state
        opp_raise = 0
        while curr.street>0:
            curr = curr.previous_state
        total_raise = curr.pips[1-active]
        while curr.previous_state:
            if curr.pips[1-active] > curr.pips[active]: # they raised
                opp_raise += curr.pips[1-active]-curr.pips[active]
            curr = curr.previous_state
        print(f'total raised {total_raise}, they raised {opp_raise}')
        return total_raise, opp_raise


    def auction_estimate(self, hand, flop, iters):
        """
        hand (list): your cards (length 2)
        flop (list): cards on the board (length 3)
        iters (int): number of monte carlo iterations

        Returns (
            probability of winning if we tie the auction (same method as preflop_estimate),
            probability of winning if we win the auction,
            probability of winning if we lose the auction
            )
        """
        deck = eval7.Deck()
        my_cards = [eval7.Card(card) for card in hand]
        flop_cards = [eval7.Card(card) for card in flop]
        for card in my_cards+flop_cards:
            deck.cards.remove(card)
        wins,wins3,wins2 = 0,0,0

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
            if val2 > opp2: # as in preflop_estimate, proxy for a tie in the auction (which would be val3 > opp3)
                wins += 1
        return wins/iters, wins3/iters, wins2/iters
    

    def train_auction_model(self):
        """
        Trains self.auction_model for use in the rest of the game

        Returns None
        """
        x,y = torch.tensor([self.my_pwins2,self.my_pwins3,self.preflop_raises,self.opp_raises]).T, torch.tensor(self.opp_bids)
        optimizer = optim.SGD(self.auction_model.parameters(), lr=0.01)
        for epoch in range(10):
            yhat = self.auction_model(x)
            loss = self.auction_loss(yhat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch}, Loss: {loss.item()}')


    def auction_loss(self,yhat,y):
        return torch.abs(torch.sub(yhat,y)).sum()
        

    def round_estimate(self, hand, opp, board, iters):
        """
        hand (list): your cards (length 2 or 3)
        opp (int): number of cards your opponnet has (2 or 3)
        board (list): cards on the board (length 3, 4, or 5)
        iters (int): number of monte carlo iterations

        Return (probability of winning, probability of losing) given the current information
        """
        deck = eval7.Deck()
        my_cards = [eval7.Card(card) for card in hand]
        flipped_cards = [eval7.Card(card) for card in board]
        for card in my_cards+flipped_cards:
            deck.cards.remove(card)
        wins,losses,ties = 0,0,0

        for i in range(iters):
            deck.shuffle()
            unflipped = 5-len(board)
            board_cards = flipped_cards + deck[:unflipped]
            opp_cards = deck[unflipped:unflipped+2] if opp==2 else deck[unflipped:unflipped+3]
            my_val = eval7.evaluate(my_cards+board_cards)
            opp_val = eval7.evaluate(opp_cards+board_cards)
            if my_val > opp_val:
                wins +=1
            elif my_val < opp_val:
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
                if FoldAction in legal_actions:
                    self.preflops += 1
                self.p_win = self.preflop_estimate(my_cards, 150)
                if game_state.round_num > 200:
                    if self.opp_folds/self.opp_preflops > self.folds/self.preflops:
                        self.cutoff += 0.01
                    elif self.cutoff > 0.55:
                        self.cutoff -= 0.01
                if self.p_win < self.cutoff and FoldAction in legal_actions:
                    if round_state.button==0:
                        self.opp_preflop_opportunity = False
                    self.folds += 1
                    self.folded = True
                    return FoldAction()
            if self.p_win >= self.cutoff and RaiseAction in legal_actions and round_state.button<2:
                if min_raise == 400:
                    return FoldAction()
                else:
                    # print(min_raise,min_raise+int((self.p_win-0.5)*(max_raise-min_raise)),max_raise)
                    return RaiseAction(random.randint(min_raise,min_raise+int((self.p_win-0.4)*(max_raise-min_raise))))
            elif CheckAction in legal_actions:
                return CheckAction()
            else:
                return CallAction()

        # auction
        elif BidAction in legal_actions:
            max_bid = opp_stack+1 if my_stack > opp_stack else my_stack
            self.p_win,self.p_win3,self.p_win2 = self.auction_estimate(my_cards, board_cards, 150)
            if self.p_win < 0.35:
                return BidAction(0)
            elif self.p_win > 0.65:
                return BidAction(max_bid)
            elif game_state.round_num <= 0.75*NUM_ROUNDS:
                auction_val = int((0.5+self.p_win3-self.p_win2)*max_bid)
                auction_val = max(auction_val,0)
                auction_val = min(auction_val,max_bid)
                print('auction', auction_val, max_bid)
                return BidAction(auction_val)
            else:
                total_raise,opp_raise = self.get_preflop_raises(round_state,active)
                auction_val = self.auction_model(torch.tensor([self.p_win2,self.p_win3,total_raise,opp_raise])).item()
                print('auction nn', auction_val)
                auction_val = max(auction_val,0)
                auction_val = min(auction_val,max_bid)
                return BidAction(auction_val)

        # normal round
        opp = 2 if my_bid > opp_bid else 3
        p_win,p_lose = self.round_estimate(my_cards,opp,board_cards,150)
        if p_win*opp_contribution - p_lose*(my_contribution+continue_cost) < -1*my_contribution:
            if CheckAction in legal_actions:
                return CheckAction()
            return FoldAction()
        if RaiseAction in legal_actions and (random.random()+p_win) > 0.8:
            raise_amt = int((p_win-p_lose)*effective_stack)
            print('normal round', min_raise, raise_amt, max_raise)
            raise_amt = min(raise_amt,max_raise)
            if street < 4:
                raise_amt = raise_amt//2
            if raise_amt < min_raise+1:
                return RaiseAction(min_raise)
            else:
                return RaiseAction(random.randint(min_raise,raise_amt))
        if CheckAction in legal_actions:
            return CheckAction()
        return CallAction()


if __name__ == "__main__":
    run_bot(Player(), parse_args())
