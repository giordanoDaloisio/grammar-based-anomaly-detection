class Nonterminal:

    def __init__(self, symbol):
        self.symbol = symbol


class Production:
    def __init__(self, lhs, rhs):
        #lhs is a Nonterminal
        #assert(isinstance(lhs, Nonterminal))
        self.lhs = lhs
        self.rhs = rhs


class Grammar:
    def __init__(self, productions, start_symbol):
        self.productions = productions
        self.start_symbol = start_symbol

        self.nonterminals, self.terminals = self.gather_symbols()


    def gather_symbols(self):
        nonterminals = set()
        terminals = set()
        for production in self.productions:
            nonterminals.add(production.lhs)
            for symbol in production.rhs:
                if isinstance(symbol, Nonterminal):
                    nonterminals.add(symbol)
                else:
                    terminals.add(symbol)
        return nonterminals, terminals


    def add_production(self, production):
        if production not in self.productions:
            self.productions.append(production)
        else:
            existing_prod = self.productions[self.productions.index(production)]
            return existing_prod

        self.nonterminals, self.terminals = self.gather_symbols()
        return 0


        