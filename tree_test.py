from nltk import Tree
from nltk.draw.util import CanvasFrame
from nltk.draw import TreeWidget

cf = CanvasFrame()
t = Tree('(S  (CH_POS_PREPOST move)  (PRE_POST    (PRE      (the the)      (_entity (F_HSV green) (F_SHAPE sphere)))    (PREPOST_connect (to to) (the the))    (POST      (_F_POS (F_POS (_bottom_left (bottom bottom) (left left)))) (corner corner))))')
tc = TreeWidget(cf.canvas(),t)
cf.add_widget(tc,10,10) # (10,10) offsets
cf.print_to_file('/home/omari/Dropbox/robot_modified/trees/test.ps')
cf.destroy()
