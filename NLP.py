from nltk import PCFG


class NLP():
    
    def __init__(self):
        pass
    
    #--------------------------------------------------------------------------------------------------------#
    def _build_parser(self,hypotheses,sentences):
        self._update_terminals(hypotheses)
        self._update_nonterminals(sentences)
        self._build_PCFG()
        
    #--------------------------------------------------------------------------------------------------------#
    def _update_terminals(self,hypotheses):
        self.grammar = ''
        self.F = {}
        self.F['features'] = {}
        self.F['sum'] = {}
        for word in hypotheses:
            for feature in hypotheses[word]:
                if feature not in ['possibilities','all']:
                    if feature not in self.F['features']:
                        self.F['features'][feature] = []
                        self.F['sum'][feature] = 0
                    for hyp in hypotheses[word][feature]:
                        self.F['features'][feature].append((word,hyp[1]))
                        self.F['sum'][feature] += hyp[1]
                        
        for feature in self.F['features']:
            l = len(self.F['features'][feature])
            for hyp in self.F['features'][feature]:
                self.grammar += feature+" -> '"+hyp[0]+"' ["+str(hyp[1]/self.F['sum'][feature])+"]"+'\n'
                
    #--------------------------------------------------------------------------------------------------------#
    def _update_nonterminals(self,S):
        for s in S:
            sentence = S[s].split(' ')
            indices = {}
            for feature in self.F['features']:
                indices[feature] = []
                for hyp in self.F['features'][feature]:
                    A = [i for i, x in enumerate(sentence) if x == hyp[0]]
                    for i in A:
                        indices[feature].append(i)
            print indices
    
    #--------------------------------------------------------------------------------------------------------#
    def _build_PCFG(self):
        if self.grammar != '':
            pcfg1 = PCFG.fromstring(self.grammar)
            print pcfg1
        
        
        
        
        
