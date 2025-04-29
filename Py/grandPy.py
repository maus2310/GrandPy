
# Wie ich es verstehe sollte das GrandR Object konzeptionell ca. so aussehen
# (damit der Code Sinn macht parralell die Funktion grandR aus grandR.R anschauen):

class GrandPy:
    def __init__(self, prefix=None, gene_info=None, slots=None, coldata=None, metadata=None, analyses=None, plots=None, parent=None):
        if parent is not None:      # Falls ich das mit dem Parent richtig interpretiere
            if prefix is None:
                self.prefix = parent.prefix
            if gene_info is None:
                self.gene_info = parent.gene_info
            if slots is None:
                self.slots = parent.data
            if coldata is None:
                self.coldata = parent.coldata
            if metadata is None:
                self.metadata = parent.metadata
        self.analyses = analyses
        self.plots = plots

    # Hier fehlt noch:
    # def checknames(): Überprüft ob Datenstruktur Sinn macht
    # 1 for und 3 ifs

# Bei dieser Definition sind 2 große Unterschied zu R:
#   1. Hier ist es eine Klasse, keine Funktion
#   2. Hier wird kein Object der Klasse zurückgegeben



# alternativ lässt es sich (wie in R) auch als eine Funktion schreiben:

def grandPy(prefix=None, gene_info=None, slots=None, coldata=None, metadata=None, analyses=None, plots=None, parent=None):
    if parent is not None:
        if prefix is None:
            prefix = parent.prefix
        if gene_info is None:
            gene_info = parent.gene_info
        if slots is None:
            slots = parent.data
        if coldata is None:
            coldata = parent.coldata
        if metadata is None:
            metadata = parent.metadata


    # Der ganze Rest fehlt hier auch


    info = type("grandPy", (dict,), {})()   # Erstellt ein (noch leeres) Object der Klasse grandPy das identisch zu einem Dictonary ist
    info["prefix"] = prefix
    info["gene.info"] = gene_info
    info["data"] = slots
    info["coldata"] = coldata
    info["metadata"] = metadata
    info["analysis"] = analyses
    info["plots"] = plots

    return info