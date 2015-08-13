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
    from nltk.draw.tree import draw_trees
    from nltk import Tree
    from nltk.draw.util import CanvasFrame
    from nltk.draw import TreeWidget

    # Define two demos.  Each demo has a sentence and a grammar.
    # demos = [('move the green sphere to the bottom left corner', learned_pcfg),
    #          ('move the green ball over the red block', learned_pcfg),
    #          ('take the green pyramid and put it in the top left corner', learned_pcfg),
    #           ('put the green pyramid on the red block', learned_pcfg),
    #           ('move the red cylinder and place it on top of the blue cylinder that is on top of a green cylinder', learned_pcfg),]

    # Ask the user which demo they want to use.
    # print()
    # for i in range(len(demos)):
    #     print('%3s: %s' % (i+1, demos[i][0]))
    #     print('     %r' % demos[i][1])
    #     print()
    # print('Which demo (%d-%d)? ' % (1, len(demos)), end=' ')
    # try:
    #     snum = int(sys.stdin.readline().strip())-1
    #     sent, grammar = demos[snum]
    # except:
    #     print('Bad sentence number')
    #     return

    max_scene = 61

    if max_scene<10:            sc = '0000'+str(max_scene)
    elif max_scene<100:         sc = '000'+str(max_scene)
    elif max_scene<1000:        sc = '00'+str(max_scene)
    elif max_scene<10000:       sc = '0'+str(max_scene)

    g = 'grammar_'+sc+'.txt'
    file1 = open('/home/omari/Dropbox/robot_modified/grammar/'+g, 'r')
    grammar = ''
    g1 = [i for i in file1.readlines()]
    for i in g1:
        grammar += i
    learned_pcfg = PCFG.fromstring(grammar)
    grammar = learned_pcfg

    file1 = open('/home/omari/Dropbox/robot_modified/hypotheses/matched_commands.txt', 'r')
    g1 = [i for i in file1.readlines()]
    for line in g1:
        sent = line.split('\n')[0].split('-')[-1]
        scene = line.split('\n')[0].split('-')[0]
        sent_num = line.split('\n')[0].split('-')[1]
        print(line)


        # Tokenize the sentence.
        tokens = sent.split()

        parser = ViterbiParser(grammar)
        all_parses = {}

        # print('\nsent: %s\nparser: %s\ngrammar: %s' % (sent,parser,grammar))
        parser.trace(3)
        parses = parser.parse_all(tokens)
        average = (reduce(lambda a,b:a+b.prob(), parses, 0)/len(parses)
                   if parses else 0)
        num_parses = len(parses)
        for p in parses:
            all_parses[p.freeze()] = 1

        # Print some summary statistics
        # print()
        # print('Time (secs)   # Parses   Average P(parse)')
        # print('-----------------------------------------')
        # print('%11.4f%11d%19.14f' % (time, num_parses, average))
        parses = all_parses.keys()
        if parses:
            p = reduce(lambda a,b:a+b.prob(), parses, 0)/len(parses)
        else: p = 0
        # print('------------------------------------------')
        # print('%11s%11d%19.14f' % ('n/a', len(parses), p))

        # Ask the user if we should draw the parses.
        # print()
        # print('Draw parses (y/n)? ', end=' ')
        # if sys.stdin.readline().strip().lower().startswith('y'):

        #     print('  please wait...')
        # draw_trees(*parses)

        cf = CanvasFrame()
        # t = Tree(parses)
        t = Tree.fromstring('(S  (CH_POS_PREPOST move)  (PRE_POST    (PRE      (the the)      (_entity (F_HSV green) (F_SHAPE sphere)))    (PREPOST_connect (to to) (the the))    (POST      (_F_POS (F_POS (_bottom_left (bottom bottom) (left left)))) (corner corner))))')

        tc = TreeWidget(cf.canvas(), t, draggable=1,
                        node_font=('helvetica', -14),
                        leaf_font=('helvetica', -12),
                        roof_fill='white', roof_color='black',
                        leaf_color='green4', node_color='blue4')
        cf.add_widget(tc,10,10)

        # tc = TreeWidget(cf.canvas(),t)
        # cf.add_widget(tc,10,10) # (10,10) offsets
        cf.print_to_file('/home/omari/Dropbox/robot_modified/trees/scene-'+scene+'-'+sent_num+'.ps')
        cf.destroy()



        # Ask the user if we should print the parses.
        # print()
        # print(parses)
        # for parse in parses:
        #     print(parse)

if __name__ == '__main__':
    demo()
