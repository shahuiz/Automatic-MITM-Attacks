    f.write('\\begin{scope}[yshift = %d cm, xshift = %d cm]\n\n'   %(-r*(3*NROW+y_shift), slot*(NCOL+x_shift))) 
            f.write('\\draw[edge, %s] (%d,%d) -- (%d,%d) -- (%d,%d) -- (%d,%d) -- (%d,%d);\n'    
            %(arrow, 
            0.5*NCOL, 0, 
            0.5*NCOL, -NROW-y_shift, 
            NCOL-(slot+1)*(x_shift+NCOL), -NROW-y_shift, 
            NCOL-(slot+1)*(x_shift+NCOL), -1.5*NROW-1.5*y_shift, 
            -slot*(x_shift+NCOL), -1.5*NROW-1.5*y_shift,))
            f.write('\n'+'\\end{scope}'+'\n')
            f.write('\n\n')