# Natural Language Toolkit: Viterbi Probabilistic Parser
#
# Copyright (C) 2001-2015 NLTK Project
# Author: Edward Loper <edloper@gmail.com>
#         Steven Bird <stevenbird1@gmail.com>
# URL: <http://nltk.org/>
# For license information, see LICENSE.TXT
from __future__ import print_function, unicode_literals

from functools import reduce
from nltk.tree import Tree, ProbabilisticTree
from nltk.compat import python_2_unicode_compatible

from nltk.parse.api import ParserI
from nltk import PCFG


toy_pcfg4 = PCFG.fromstring("""
S -> _S [1.0]
_S_connect -> and [1.0]
TV -> it _relation [1.0]
_S -> _S _S_connect _S [0.333333]
_S -> motion_TV TV [0.333333]
_S -> motion_TE TE [0.333333]
_on_top_of -> on top of [1.0]
_entity -> color shape [0.75]
_entity -> _entity that is _relation [0.25]
TE -> the _entity [1.0]
_relation -> direction [0.5]
_relation -> _relation the _entity [0.25]
_relation -> _relation a _entity [0.25]
and -> 'and' [1.0]
on -> 'on' [1.0]
that -> 'that' [1.0]
of -> 'of' [1.0]
top -> 'top' [1.0]
it -> 'it' [1.0]
a -> 'a' [1.0]
the -> 'the' [1.0]
is -> 'is' [1.0]
color -> 'blue' [0.332638]
color -> 'green' [0.337461]
color -> 'red' [0.329901]
shape -> 'cylinder' [1.0]
direction -> _on_top_of [1.0]
motion_TV -> 'place' [1.0]
motion_TE -> 'move' [1.0]
""")

toy_pcfg5 = PCFG.fromstring("""
S -> motion_TETV TE_TV [1.0]
_ball_to -> ball to [1.0]
_over_the -> over the [1.0]
_location -> location [1.0]
TETV_connect -> in the [0.444444]
TETV_connect -> to the [0.222222]
TETV_connect -> on the [0.111111]
TETV_connect -> the [0.0740741]
TETV_connect -> green [0.0740741]
TETV_connect -> on [0.0740741]
_bottom_left -> bottom left [1.0]
TV -> _location corner [0.685714]
TV -> _relation [0.314286]
TE_TV -> TE TETV_connect TV [0.771429]
TE_TV -> TE TV [0.228571]
_on_top_of -> on top of [1.0]
_top_of_the -> top of the [1.0]
_entity -> color shape [0.673913]
_entity -> shape [0.173913]
_entity -> color [0.0869565]
_entity -> shape color [0.0652174]
_on_the -> on the [1.0]
_the_green -> the green [1.0]
TE -> the _entity [0.542857]
TE -> _entity [0.371429]
TE -> the green _entity [0.0857143]
_left_back -> left back [1.0]
_relation -> direction [0.5]
_relation -> _relation _entity [0.272727]
_relation -> _relation top of _entity [0.181818]
_relation -> _relation the _entity [0.0454545]
_to_the -> to the [1.0]
on -> 'on' [1.0]
ball -> 'ball' [1.0]
bottom -> 'bottom' [1.0]
of -> 'of' [1.0]
top -> 'top' [1.0]
back -> 'back' [1.0]
to -> 'to' [1.0]
green -> 'green' [1.0]
in -> 'in' [1.0]
corner -> 'corner' [1.0]
the -> 'the' [1.0]
over -> 'over' [1.0]
left -> 'left' [1.0]
color -> 'to' [0.0409916]
color -> 'green' [0.300108]
color -> _the_green [0.301532]
color -> 'the' [0.0435557]
color -> 'red' [0.293345]
color -> _to_the [0.0204672]
shape -> 'sphere' [0.405428]
shape -> 'cube' [0.119889]
shape -> 'ball' [0.330799]
shape -> 'block' [0.143883]
direction -> _ball_to [0.349116]
direction -> _over_the [0.188216]
direction -> _on_the [0.178807]
direction -> _top_of_the [0.189182]
direction -> _on_top_of [0.0946782]
motion_TETV -> 'move' [0.838624]
motion_TETV -> 'place' [0.161376]
location -> _left_back [0.125076]
location -> _bottom_left [0.874924]
""")


toy_pcfg3 = PCFG.fromstring("""
S -> motion_TETV TE_TV [0.838709677419]
S -> _S [0.161290322581]
_location -> location [1.0]
TE_TV -> TE TETV_connect TV [0.769230769231]
TE_TV -> TE TV [0.230769230769]
_shape -> shape [1.0]
_top_left -> top left [1.0]
_entity -> color shape [1.0]
_place_on_the -> place on the [1.0]
_sphere_on_the -> sphere on the [1.0]
_direction -> direction [1.0]
TV -> _location corner [0.614035087719]
TV -> _direction _entity [0.19298245614]
TV -> _direction the _entity [0.0526315789474]
TV -> _location left corner [0.0526315789474]
TV -> _direction _color brick [0.0175438596491]
TV -> _direction of _entity [0.0175438596491]
TV -> it _direction _entity [0.0175438596491]
TV -> it _location left corner [0.0175438596491]
TV -> it in the _location corner [0.0175438596491]
_left_back -> left back [1.0]
_and_place_on -> and place on [1.0]
_it_on_the -> it on the [1.0]
TE -> the _entity [0.543859649123]
TE -> the _shape [0.210526315789]
TE -> _color [0.0701754385965]
TE -> the _entity and [0.0526315789474]
TE -> _entity and [0.0350877192982]
TE -> _entity [0.0350877192982]
TE -> _shape [0.0350877192982]
TE -> the _color [0.0175438596491]
_top_of_the -> top of the [1.0]
TETV_connect -> in the [0.375]
TETV_connect -> to the [0.225]
TETV_connect -> on the [0.15]
TETV_connect -> on [0.15]
TETV_connect -> the [0.05]
TETV_connect -> and [0.05]
_bottom_left -> bottom left [1.0]
_over_the -> over the [1.0]
_S -> _S _S [0.5]
_S -> motion_TV TV [0.25]
_S -> motion_TE TE [0.25]
_the_sphere_on -> the sphere on [1.0]
_on_top_of -> on top of [1.0]
_in_the_top -> in the top [1.0]
_it_on_top -> it on top [1.0]
_the_prism_on -> the prism on [1.0]
_color -> color [1.0]
and -> 'and' [1.0]
on -> 'on' [1.0]
to -> 'to' [1.0]
bottom -> 'bottom' [1.0]
of -> 'of' [1.0]
top -> 'top' [1.0]
back -> 'back' [1.0]
it -> 'it' [1.0]
sphere -> 'sphere' [1.0]
prism -> 'prism' [1.0]
place -> 'place' [1.0]
in -> 'in' [1.0]
corner -> 'corner' [1.0]
the -> 'the' [1.0]
brick -> 'brick' [1.0]
over -> 'over' [1.0]
left -> 'left' [1.0]
direction -> _it_on_top [0.102140081741]
direction -> _over_the [0.0509094184834]
direction -> _top_of_the [0.307005334325]
direction -> _on_top_of [0.307288879649]
direction -> _and_place_on [0.0486657306874]
direction -> _place_on_the [0.0830739978875]
direction -> _it_on_the [0.100916557226]
motion_TE -> 'put' [0.300781820935]
motion_TE -> 'place' [0.201059124714]
motion_TE -> 'take' [0.197252863919]
motion_TE -> 'pick' [0.300906190432]
color -> _the_prism_on [0.027545098561]
color -> _the_sphere_on [0.0276830559885]
color -> 'black' [0.238976229457]
color -> 'green' [0.632692697002]
color -> _sphere_on_the [0.0135413637727]
color -> 'red' [0.0595615552188]
motion_TETV -> 'move' [0.795694460495]
motion_TETV -> 'place' [0.146014373377]
motion_TETV -> 'pick' [0.0582911661273]
shape -> 'cylinder' [0.106159686144]
shape -> 'ball' [0.0786873106762]
shape -> 'tetrahedron' [0.0514335865582]
shape -> 'sphere' [0.0793334863204]
shape -> 'prism' [0.158078748524]
shape -> 'pyramid' [0.380257632003]
shape -> 'can' [0.146049549774]
location -> _left_back [0.142604687771]
location -> _top_left [0.524434701318]
location -> _bottom_left [0.215038473725]
location -> _in_the_top [0.117922137186]
motion_TV -> 'put' [0.300781820935]
motion_TV -> 'place' [0.201059124714]
motion_TV -> 'take' [0.197252863919]
motion_TV -> 'pick' [0.300906190432]
    """)

##//////////////////////////////////////////////////////
##  Viterbi PCFG Parser
##//////////////////////////////////////////////////////

@python_2_unicode_compatible
class ViterbiParser(ParserI):
    """
    A bottom-up ``PCFG`` parser that uses dynamic programming to find
    the single most likely parse for a text.  The ``ViterbiParser`` parser
    parses texts by filling in a "most likely constituent table".
    This table records the most probable tree representation for any
    given span and node value.  In particular, it has an entry for
    every start index, end index, and node value, recording the most
    likely subtree that spans from the start index to the end index,
    and has the given node value.

    The ``ViterbiParser`` parser fills in this table incrementally.  It starts
    by filling in all entries for constituents that span one element
    of text (i.e., entries where the end index is one greater than the
    start index).  After it has filled in all table entries for
    constituents that span one element of text, it fills in the
    entries for constitutants that span two elements of text.  It
    continues filling in the entries for constituents spanning larger
    and larger portions of the text, until the entire table has been
    filled.  Finally, it returns the table entry for a constituent
    spanning the entire text, whose node value is the grammar's start
    symbol.

    In order to find the most likely constituent with a given span and
    node value, the ``ViterbiParser`` parser considers all productions that
    could produce that node value.  For each production, it finds all
    children that collectively cover the span and have the node values
    specified by the production's right hand side.  If the probability
    of the tree formed by applying the production to the children is
    greater than the probability of the current entry in the table,
    then the table is updated with this new tree.

    A pseudo-code description of the algorithm used by
    ``ViterbiParser`` is:

    | Create an empty most likely constituent table, *MLC*.
    | For width in 1...len(text):
    |   For start in 1...len(text)-width:
    |     For prod in grammar.productions:
    |       For each sequence of subtrees [t[1], t[2], ..., t[n]] in MLC,
    |         where t[i].label()==prod.rhs[i],
    |         and the sequence covers [start:start+width]:
    |           old_p = MLC[start, start+width, prod.lhs]
    |           new_p = P(t[1])P(t[1])...P(t[n])P(prod)
    |           if new_p > old_p:
    |             new_tree = Tree(prod.lhs, t[1], t[2], ..., t[n])
    |             MLC[start, start+width, prod.lhs] = new_tree
    | Return MLC[0, len(text), start_symbol]

    :type _grammar: PCFG
    :ivar _grammar: The grammar used to parse sentences.
    :type _trace: int
    :ivar _trace: The level of tracing output that should be generated
        when parsing a text.
    """
    def __init__(self, grammar, trace=0):
        """
        Create a new ``ViterbiParser`` parser, that uses ``grammar`` to
        parse texts.

        :type grammar: PCFG
        :param grammar: The grammar used to parse texts.
        :type trace: int
        :param trace: The level of tracing that should be used when
            parsing a text.  ``0`` will generate no tracing output;
            and higher numbers will produce more verbose tracing
            output.
        """
        self._grammar = grammar
        self._trace = trace

    def grammar(self):
        return self._grammar

    def trace(self, trace=2):
        """
        Set the level of tracing output that should be generated when
        parsing a text.

        :type trace: int
        :param trace: The trace level.  A trace level of ``0`` will
            generate no tracing output; and higher trace levels will
            produce more verbose tracing output.
        :rtype: None
        """
        self._trace = trace

    def parse(self, tokens):
        # Inherit docs from ParserI

        tokens = list(tokens)
        self._grammar.check_coverage(tokens)

        # The most likely constituent table.  This table specifies the
        # most likely constituent for a given span and type.
        # Constituents can be either Trees or tokens.  For Trees,
        # the "type" is the Nonterminal for the tree's root node
        # value.  For Tokens, the "type" is the token's type.
        # The table is stored as a dictionary, since it is sparse.
        constituents = {}

        # Initialize the constituents dictionary with the words from
        # the text.
        if self._trace: print(('Inserting tokens into the most likely'+
                               ' constituents table...'))
        for index in range(len(tokens)):
            token = tokens[index]
            constituents[index,index+1,token] = token
            if self._trace > 1:
                self._trace_lexical_insertion(token, index, len(tokens))

        # Consider each span of length 1, 2, ..., n; and add any trees
        # that might cover that span to the constituents dictionary.
        for length in range(1, len(tokens)+1):
            if self._trace:
                print(('Finding the most likely constituents'+
                       ' spanning %d text elements...' % length))
            for start in range(len(tokens)-length+1):
                span = (start, start+length)
                self._add_constituents_spanning(span, constituents,
                                                tokens)

        # Return the tree that spans the entire text & have the right cat
        tree = constituents.get((0, len(tokens), self._grammar.start()))
        if tree is not None:
            yield tree

    def _add_constituents_spanning(self, span, constituents, tokens):
        """
        Find any constituents that might cover ``span``, and add them
        to the most likely constituents table.

        :rtype: None
        :type span: tuple(int, int)
        :param span: The section of the text for which we are
            trying to find possible constituents.  The span is
            specified as a pair of integers, where the first integer
            is the index of the first token that should be included in
            the constituent; and the second integer is the index of
            the first token that should not be included in the
            constituent.  I.e., the constituent should cover
            ``text[span[0]:span[1]]``, where ``text`` is the text
            that we are parsing.

        :type constituents: dict(tuple(int,int,Nonterminal) -> ProbabilisticToken or ProbabilisticTree)
        :param constituents: The most likely constituents table.  This
            table records the most probable tree representation for
            any given span and node value.  In particular,
            ``constituents(s,e,nv)`` is the most likely
            ``ProbabilisticTree`` that covers ``text[s:e]``
            and has a node value ``nv.symbol()``, where ``text``
            is the text that we are parsing.  When
            ``_add_constituents_spanning`` is called, ``constituents``
            should contain all possible constituents that are shorter
            than ``span``.

        :type tokens: list of tokens
        :param tokens: The text we are parsing.  This is only used for
            trace output.
        """
        # Since some of the grammar productions may be unary, we need to
        # repeatedly try all of the productions until none of them add any
        # new constituents.
        changed = True
        while changed:
            changed = False

            # Find all ways instantiations of the grammar productions that
            # cover the span.
            instantiations = self._find_instantiations(span, constituents)

            # For each production instantiation, add a new
            # ProbabilisticTree whose probability is the product
            # of the childrens' probabilities and the production's
            # probability.
            for (production, children) in instantiations:
                subtrees = [c for c in children if isinstance(c, Tree)]
                p = reduce(lambda pr,t:pr*t.prob(),
                           subtrees, production.prob())
                node = production.lhs().symbol()
                tree = ProbabilisticTree(node, children, prob=p)

                # If it's new a constituent, then add it to the
                # constituents dictionary.
                c = constituents.get((span[0], span[1], production.lhs()))
                if self._trace > 1:
                    if c is None or c != tree:
                        if c is None or c.prob() < tree.prob():
                            print('   Insert:', end=' ')
                        else:
                            print('  Discard:', end=' ')
                        self._trace_production(production, p, span, len(tokens))
                if c is None or c.prob() < tree.prob():
                    constituents[span[0], span[1], production.lhs()] = tree
                    changed = True

    def _find_instantiations(self, span, constituents):
        """
        :return: a list of the production instantiations that cover a
            given span of the text.  A "production instantiation" is
            a tuple containing a production and a list of children,
            where the production's right hand side matches the list of
            children; and the children cover ``span``.  :rtype: list
            of ``pair`` of ``Production``, (list of
            (``ProbabilisticTree`` or token.

        :type span: tuple(int, int)
        :param span: The section of the text for which we are
            trying to find production instantiations.  The span is
            specified as a pair of integers, where the first integer
            is the index of the first token that should be covered by
            the production instantiation; and the second integer is
            the index of the first token that should not be covered by
            the production instantiation.
        :type constituents: dict(tuple(int,int,Nonterminal) -> ProbabilisticToken or ProbabilisticTree)
        :param constituents: The most likely constituents table.  This
            table records the most probable tree representation for
            any given span and node value.  See the module
            documentation for more information.
        """
        rv = []
        for production in self._grammar.productions():
            childlists = self._match_rhs(production.rhs(), span, constituents)

            for childlist in childlists:
                rv.append( (production, childlist) )
        return rv

    def _match_rhs(self, rhs, span, constituents):
        """
        :return: a set of all the lists of children that cover ``span``
            and that match ``rhs``.
        :rtype: list(list(ProbabilisticTree or token)

        :type rhs: list(Nonterminal or any)
        :param rhs: The list specifying what kinds of children need to
            cover ``span``.  Each nonterminal in ``rhs`` specifies
            that the corresponding child should be a tree whose node
            value is that nonterminal's symbol.  Each terminal in ``rhs``
            specifies that the corresponding child should be a token
            whose type is that terminal.
        :type span: tuple(int, int)
        :param span: The section of the text for which we are
            trying to find child lists.  The span is specified as a
            pair of integers, where the first integer is the index of
            the first token that should be covered by the child list;
            and the second integer is the index of the first token
            that should not be covered by the child list.
        :type constituents: dict(tuple(int,int,Nonterminal) -> ProbabilisticToken or ProbabilisticTree)
        :param constituents: The most likely constituents table.  This
            table records the most probable tree representation for
            any given span and node value.  See the module
            documentation for more information.
        """
        (start, end) = span

        # Base case
        if start >= end and rhs == (): return [[]]
        if start >= end or rhs == (): return []

        # Find everything that matches the 1st symbol of the RHS
        childlists = []
        for split in range(start, end+1):
            l=constituents.get((start,split,rhs[0]))
            if l is not None:
                rights = self._match_rhs(rhs[1:], (split,end), constituents)
                childlists += [[l]+r for r in rights]

        return childlists

    def _trace_production(self, production, p, span, width):
        """
        Print trace output indicating that a given production has been
        applied at a given location.

        :param production: The production that has been applied
        :type production: Production
        :param p: The probability of the tree produced by the production.
        :type p: float
        :param span: The span of the production
        :type span: tuple
        :rtype: None
        """

        str = '|' + '.' * span[0]
        str += '=' * (span[1] - span[0])
        str += '.' * (width - span[1]) + '| '
        str += '%s' % production
        if self._trace > 2: str = '%-40s %12.10f ' % (str, p)

        print(str)

    def _trace_lexical_insertion(self, token, index, width):
        str = '   Insert: |' + '.' * index + '=' + '.' * (width-index-1) + '| '
        str += '%s' % (token,)
        print(str)

    def __repr__(self):
        return '<ViterbiParser for %r>' % self._grammar


##//////////////////////////////////////////////////////
##  Test Code
##//////////////////////////////////////////////////////
def demo():
    """
    A demonstration of the probabilistic parsers.  The user is
    prompted to select which demo to run, and how many parses should
    be found; and then each parser is run on the same demo, and a
    summary of the results are displayed.
    """
    import sys, time
    from nltk import tokenize
    from nltk.parse import ViterbiParser
    from nltk.grammar import toy_pcfg1, toy_pcfg2

    # Define two demos.  Each demo has a sentence and a grammar.
    demos = [('move the green sphere to the bottom left corner', toy_pcfg5),
             ('move the green ball over the red block', toy_pcfg5),
             ('take the green pyramid and put it in the top left corner', toy_pcfg3),
              ('put the green pyramid on the red block', toy_pcfg3),
              ('move the red cylinder and place it on top of the blue cylinder that is on top of a green cylinder', toy_pcfg4),]

    # Ask the user which demo they want to use.
    print()
    for i in range(len(demos)):
        print('%3s: %s' % (i+1, demos[i][0]))
        print('     %r' % demos[i][1])
        print()
    print('Which demo (%d-%d)? ' % (1, len(demos)), end=' ')
    try:
        snum = int(sys.stdin.readline().strip())-1
        sent, grammar = demos[snum]
    except:
        print('Bad sentence number')
        return

    # Tokenize the sentence.
    tokens = sent.split()

    parser = ViterbiParser(grammar)
    all_parses = {}

    print('\nsent: %s\nparser: %s\ngrammar: %s' % (sent,parser,grammar))
    parser.trace(3)
    t = time.time()
    parses = parser.parse_all(tokens)
    time = time.time()-t
    average = (reduce(lambda a,b:a+b.prob(), parses, 0)/len(parses)
               if parses else 0)
    num_parses = len(parses)
    for p in parses:
        all_parses[p.freeze()] = 1

    # Print some summary statistics
    print()
    print('Time (secs)   # Parses   Average P(parse)')
    print('-----------------------------------------')
    print('%11.4f%11d%19.14f' % (time, num_parses, average))
    parses = all_parses.keys()
    if parses:
        p = reduce(lambda a,b:a+b.prob(), parses, 0)/len(parses)
    else: p = 0
    print('------------------------------------------')
    print('%11s%11d%19.14f' % ('n/a', len(parses), p))

    # Ask the user if we should draw the parses.
    print()
    print('Draw parses (y/n)? ', end=' ')
    if sys.stdin.readline().strip().lower().startswith('y'):
        from nltk.draw.tree import draw_trees
        print('  please wait...')
        draw_trees(*parses)

    # Ask the user if we should print the parses.
    print()
    print('Print parses (y/n)? ', end=' ')
    if sys.stdin.readline().strip().lower().startswith('y'):
        for parse in parses:
            print(parse)

if __name__ == '__main__':
    demo()
