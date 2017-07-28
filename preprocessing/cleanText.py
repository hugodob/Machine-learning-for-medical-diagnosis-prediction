import re
import string
import numpy
import pandas as pd
import os.path
import sys
import nltk
import json
import re


#to_keep_stopword=['any','few','more','not','most','no','nor','only','but','off','do',"if","or","because","as","until","while","of","at","by","for","with","about","against","between","into","through","during","before","after","above","below","to","from","up","down","in","out","on","over","under","again","further","then","once","here","there","when","where","why","how","all","both","each","other","some","such","own","same","so","than","too","very","should","now"]
stopword_set=set(["i","me","my","myself","we","our","ours","ourselves","you","your","yours","yourself","yourselves","he","him","his","himself","she","her","hers","herself","it","its","itself","they","them","their","theirs","themselves","what","which","who","whom","this","that","these","those","am","is","are","was","were","be","been","being","have","has","had","having","do","does","did","doing","a","an","the","and","but","if","or","because","as","until","while","of","at","by","for","with","about","against","between","into","through","during","before","after","above","below","to","from","up","down","in","out","on","off","over","under","again","further","then","once","here","there","when","where","why","how","all","any","both","each","few","more","most","other","some","such","no","nor","not","only","own","same","so","than","too","very","s","t","can","will","just","don","should","now"])
#for word in to_keep_stopword:
#    stopword_set.remove(word)
ngram_map_dict = {}
gene_set =['ERRFI1','CCNE1','STK11','MEN1','FAM58A','AKT1','AKT2','AKT3','ARAF','RB1','IKBKE','NKX2-1','ROS1','AXL','RARA','RAD51D','BRCA1','PIK3CA','PIK3CB','MAP3K1','EIF1AX','INPP4B','WHSC1L1','GATA3','FGFR2','GLI1','CDH1','PPP6C','MYC','YAP1','BTK','NF2','CDKN2B','ERBB3','IGF1R','CDKN2A','PIK3R1','ERBB4','BARD1','IDH2','IDH1','NUP93','RET','BRD4','PMS1','MDM2','SETD2','MDM4','FGFR3','RAF1','FGFR1','MYD88','CCND1','ARID1B','ARID1A','CCND2','B2M','TCF7L2','KIT','FOXA1','PTEN','FAT1','RUNX1','WHSC1','APC','CCND3','CTCF','KDM5C','IL7R','DNMT3B','BRCA2','FOXP1','SDHC','CDKN1B','CDKN1A','RRAS2','CARM1','RIT1','PTPN11','CASP8','RICTOR','KDM5A','XPO1','MYCN','PPM1D','SRSF2','ASXL1','TSC2','RASA1','ASXL2','JUN','PIK3R3','PIK3R2','H3F3A','JAK1','MSH2','FLT1','CHEK2','CARD11','CTLA4','TCF3','STAG2','ARID2','PBRM1','RNF43','VEGFA','HRAS','RAD21','RHOA','FGF4','FGF3','PAX8','KEAP1','ETV1','EPAS1','MGA','TP53','GNAQ','ETV6','DDR2','MPL','CBL','PAK1','MAP2K2','MEF2B','SHQ1','PRDM1','NFE2L2','NSD1','CREBBP','AGO2','PDGFRB','PDGFRA','PMS2','MAP2K1','PPP2R1A','SMAD4','PIK3CD','JAK2','ATM','SMAD2','SMAD3','SMO','POLE','ATR','NTRK2','PIM1','ABL1','BRIP1','NTRK3','IKZF1','FLT3','NCOR1','TSC1','STAT3','NPM1','BCL10','FGF19','RBM10','FANCC','FANCA','HLA-B','KDM6A','HLA-A','MAPK1','FBXW7','TGFBR2','TGFBR1','FUBP1','TET1','ERCC4','TET2','RXRA','MTOR','BCOR','DUSP4','ATRX','EP300','RAD51C','RAD51B','HIST1H1C','KNSTRN','DICER1','ARID5B','SOS1','VHL','ESR1','FOXO1','MET','SHOC2','EZH2','CDK4','KDR','CDK6','RAD50','CDK8','RHEB','NTRK1','GNAS','CIC','ERBB2','ACVR1','CDKN2C','ERCC2','ERCC3','SF3B1','HNF1A','MSH6','PTCH1','CTNNB1','LATS2','LATS1','ERG','NOTCH2','MAP2K4','ELF3','SMARCA4','CEBPA','XRCC2','BCL2L11','MYOD1','AXIN1','ALK','RAD54L','NRAS','MLH1','MED12','KLF4','AURKA','AURKB','RYBP','TERT','KMT2C','KMT2B','KMT2A','DNMT3A','SMARCB1','KMT2D','SPOP','TMPRSS2','RAB35','SRC','CDK12','AR','TP53BP1','EPCAM','RAC1','KRAS','BAP1','NF1','PTPRT','SOX9','NOTCH1','NFKBIA','U2AF1','PTPRD','FGFR4','BRAF','GNA11','FOXL2','EGFR','SDHB','EWSR1','BCL2']


# Stopwords : Meaningless and frequent words
# Bigrams : We could add a step to identify useful bigrams and merge them. ex : "do not":"do_not"
# Gene_set : Genes mentionned in our dataset

def ignore_ascii(s):
    return s.decode('unicode_escape').encode('ascii','ignore')

class CleanText():
	def __init__(self, comment_doctor):
		self.comment=ignore_ascii(comment_doctor.replace("\N","/N").replace(r'\x',"/x").replace(r'\u',"/u"))
		self.sentences=nltk.sent_tokenize(self.comment)
		self.tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
		self.comment_tokenized=[i for i in self.tokenizer.tokenize(self.comment.lower() ) if i not in stopword_set]
	
	def build_bag_of_words(self):
		freq={}
		for i in self.comment_tokenized:
			if i in freq:
				freq[i]=freq[i]+1
			else:
				freq[i]=1
		return freq

	def build_bag_of_words_is_in(self):
		freq={}
		for i in self.comment_tokenized:
			freq[i]=True
		return freq

	def sentences_about_gene(self, gene_name):
		res=[]
		for i in self.sentences:
			if gene_name in i:
				res.append(i)
		return res

	def build_gene_specific_bag_of_words(self,gene_name):
		sent=self.sentences_about_gene(gene_name)
		gene_comment_tokenized=[]
		for s in sent:
			gene_comment_tokenized=gene_comment_tokenized+[i for i in self.tokenizer.tokenize(s.lower() ) if i not in stopword_set]
		freq={}
		for i in gene_comment_tokenized:
			freq[i]=True
		return freq

	def genes_mentionned(self):
		self.mentionned_genes=[]
		for i in gene_set:
			if len(self.sentences_about_gene(i))>=1:
				self.mentionned_genes.append(i)
		print self.mentionned_genes





