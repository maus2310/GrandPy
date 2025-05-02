
# Wie ich es verstehe sollte das GrandR Object konzeptionell ca. so aussehen
# (damit der Code Sinn macht parralell die Funktion grandR aus grandR.R anschauen):

class GrandPy:
    def __init__(self, prefix=None, gene_info=None, slots=None, coldata=None, metadata=None, analyses=None, plots=None, parent=None):
        self.prefix = prefix if prefix is not None else getattr(parent, 'prefix', None)
        self.gene_info = gene_info if gene_info is not None else getattr(parent, 'gene_info', None)
        self.slots = slots if slots is not None else getattr(parent, 'data', None)
        self.coldata = coldata if coldata is not None else getattr(parent, 'coldata', None)
        self.metadata = metadata if metadata is not None else getattr(parent, 'metadata', None)
        self.analysis = analyses
        self.plots = plots

    # Hier fehlt noch:
    # def checknames(): Überprüft ob Datenstruktur Sinn macht
    # 1 for und 3 ifs




# alternativ lässt es sich (wie in R) auch als eine Funktion schreiben:

def grandPy(prefix=None, gene_info=None, slots=None, coldata=None, metadata=None, analyses=None, plots=None, parent=None):
    prefix = prefix if prefix is not None else getattr(parent, 'prefix', None)
    gene_info = gene_info if gene_info is not None else getattr(parent, 'gene_info', None)
    slots = slots if slots is not None else getattr(parent, 'data', None)
    coldata = coldata if coldata is not None else getattr(parent, 'coldata', None)
    metadata = metadata if metadata is not None else getattr(parent, 'metadata', None)


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
# Mitlerweile bevorzuge ich die erste Implementierung, da ich befürchte, dass es bei der zweiten später zu Schwierigkeiten kommen könnte.
# Zuerst würde ich gerne mal mit Erhard oder den anderen beiden drüber reden