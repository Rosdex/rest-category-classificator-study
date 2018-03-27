#encoding "utf-8"
#GRAMMAR_ROOT S

S -> Adj interp (ProductName.AdjForName) Noun interp (ProductName.Name);
S -> Noun interp (ProductName.Name) Adj interp (ProductName.AdjForName);
S -> Noun interp (ProductName.Name);


