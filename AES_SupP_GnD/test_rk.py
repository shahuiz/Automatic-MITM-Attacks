def precell(cell, fwd, bwd):
    r, i, j = cell[0], cell[1], cell[2]
    if r in fwd:
        if j==0:
            pr, pi, pj = r-1, (i+1)%4, 5
        else:
            pr, pi, pj = r, i, j-1
        qr, qi, qj = r-1, i, j
        return [pr, pi, pj], [qr, qi, qj]
    elif r in bwd:
        if j==0:
            pr, pi, pj = r, (i+1)%4, 5
        else:
            pr, pi, pj = r+1, j-1
        qr, qi, qj = r+1, j
        return [pr, pi, pj], [qr, qi, qj]
    else:
        return cell

a, b = precell([8,0,0])
a1, a2 = precell(a)
b1, b2 = precell(b)
a11, a12 = precell(a1)
a21, a22 = precell(a2)
b11, b12 = precell(b1)
b21, b22 = precell(b2)
print(precell(a11), precell(a12), precell(a21), precell(a22), precell(b11), precell(b12), precell(b21), precell(b22))