from nltk import PCFG


class NLP():
    
    def __init__(self):
        pass
        
    def _build_parser(self,hypotheses):
        grammar = ''
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
                grammar += feature+" -> '"+hyp[0]+"' ["+str(hyp[1]/self.F['sum'][feature])+"]"+'\n'
                
        if grammar != '':
            pcfg1 = PCFG.fromstring(grammar)
            print pcfg1
        

toy_pcfg1 = PCFG.fromstring("""
S -> NP VP [1.0]
NP -> Det N [0.5] | NP PP [0.25] | 'John' [0.1] | 'I' [0.15]
Det -> 'the' [0.8] | 'my' [0.2]
N -> 'man' [0.5] | 'telescope' [0.5]
VP -> VP PP [0.1] | V NP [0.7] | V [0.2]
V -> 'ate' [0.35] | 'saw' [0.65]
PP -> P NP [1.0]
P -> 'with' [0.61] | 'under' [0.39]
""")
print toy_pcfg1
