\ Model x
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  0 C1 + 0 C2 + 0 C3 + 0 C4 + 0 C5
Subject To
 R0: match_case_1 = 0.75
 GC0: C4 = 1 -> C1 + 2 C2 + C3 = 1
Bounds
Binaries
 match_case_1 C1 C2 C3 C4 C5
End
