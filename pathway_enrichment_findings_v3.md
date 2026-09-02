# Pathway Enrichment Findings — Day 47, Block 2

## Synthetic self-test
PASSED: known cell-cycle positive-control gene list correctly recovered a
"cell cycle" term as top-enriched against MSigDB_Hallmark_2020, confirming
the Enrichr pipeline call itself works before real data was touched.

## Methodological note (deviation from literal Block 2 instructions)
`switching_genes_gm12878_v1.txt` and `switching_genes_k562_v1.txt` (Block 1
output) are identical gene sets by construction. Enrichment was instead run
on the two `and_origin` subsets of AND<->INC switchers specifically (n=3,690
total from Block 1) — GM12878-origin (SIMPLE_AND in GM12878, INCONSISTENT in
K562) vs K562-origin (reverse) — since that is the actual cell-line-specific
comparison implied by the Day 38 directional finding.

## Background
Tested-gene universe = genes with a valid gate call in both cell lines
(post Block 1 exclusions), mapped to 15316 unique gene
symbols. This is the background used for both enrichment calls, per
Reimand et al. 2019's warning against genome-wide background inflating
significance.

## Gene ID -> symbol mapping
- GM12878-origin switchers: 3502 Ensembl gene_id -> 3119 mapped symbols
- K562-origin switchers: 188 Ensembl gene_id -> 170 mapped symbols
(Any gap between these counts reflects genes with no gene_name annotation
available, excluded rather than guessed at — see script stdout for exact counts.)

## Results — GM12878-origin switchers
4408 terms tested across 3 libraries. 3 significant (adj. p < 0.05) shown below.

                            Term Overlap  Adjusted P-value                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    Genes
Herpes simplex virus 1 infection 133/369      5.368941e-09 ZNF671;ZNF25;ZNF605;ZNF91;ZNF813;SYK;ZNF597;OAS3;ZNF81;ZNF235;ZNF484;RBAK;ZNF436;ZNF641;RNASEL;BAD;IKBKE;ZNF530;ZNF57;ZNF333;ZNF878;ZNF778;ZNF34;ZNF510;ZNF680;ZNF251;BCL2;ZNF480;PIK3R3;ZNF17;ZNF567;ZNF563;ZNF19;ZNF222;ZNF623;C5;ZNF212;ZNF133;ZNF79;ZNF383;ZNF619;ZNF234;TNFRSF14;ZNF607;ZNF433;ZNF283;ZNF707;ZNF184;ZNF552;IRF7;CCL5;ZNF41;ZNF717;ZNF443;ZNF630;ZNF517;ZNF555;ZNF571;IRF9;ZNF559-ZNF177;ZNF33A;TRADD;ZNF790;ZNF23;ZNF316;ZNF418;ZNF799;ZNF225;HCFC2;TAP2;IFNGR1;ZNF726;ZNF169;TRAF5;ZNF514;ZNF700;ZNF181;ZNF547;ZNF8;ZNF761;ZNF543;ZNF589;ZNF14;TICAM1;ZNF823;TP53;ZNF566;ZNF785;ZFP82;ZNF674;ZNF136;ZNF441;ZNF221;ZNF200;ZNF599;ZNF845;ZNF688;ZNF77;HLA-DMA;ZNF182;ZNF783;ZNF10;ZNF85;ZNF557;ZNF550;ZNF224;ZNF30;ZNF616;ZNF44;EIF2AK3;ZNF559;ZNF519;PIK3R1;ZNF736;IFIH1;ZNF101;ZNF764;ZNF304;NFKB1;ZNF337;ZNF891;ZNF205;ZNF189;ZNF620;ZNF248;ZNF2;ZNF596;ZNF780B;ZNF786;ZNF569;ZNF417;ZNF684;ZNF20
    Cilium Assembly (GO:0060271)  63/186      2.346075e-02                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            ENKD1;FUZ;TCTN2;FAM161A;TRAPPC14;IFT80;RPGRIP1L;PIBF1;IFT27;GORAB;B9D1;RAB3IP;KIF24;IFT22;RILPL2;ARL3;KIF3A;OFD1;EHD4;SNX10;FAM149B1;CEP162;MACIR;TMEM237;IFT81;ATAT1;CC2D2B;BBIP1;RFX3;TCTN1;TMEM216;CILK1;RP2;MKKS;FBF1;TMEM67;IFT74;RPGR;ARMC9;IFT20;DNMBP;NEK1;ARL6;BBOF1;CEP83;BBS10;KIAA0753;CEP290;STK36;IFT46;TTBK2;DYNLT2B;BBS9;RAB23;ALPK1;TOGARAM1;TXNDC15;NPHP3;TBC1D32;WDR19;KIF27;TMEM80;SPAG16
Cilium Organization (GO:0044782)  59/175      3.433855e-02                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       FUZ;TCTN2;FAM161A;TRAPPC14;IFT80;PIBF1;IFT27;B9D1;RAB3IP;KIF24;IFT22;RILPL2;ARL3;KIF3A;OFD1;EHD4;SNX10;FAM161B;CEP162;MACIR;FAM149B1;TMEM237;IFT81;ATAT1;BBIP1;RFX3;TCTN1;TMEM216;CILK1;RP2;MKKS;FBF1;TMEM67;IFT74;RPGR;ARMC9;IFT20;DNMBP;NEK1;CEP78;ARL6;CCP110;CEP83;KIAA0753;STK36;CEP290;IFT46;TTBK2;DYNLT2B;BBS9;RAB23;CCDC32;ALPK1;TOGARAM1;TXNDC15;NPHP3;WDR19;KIF27;SPAG16

### Cancer-relevant terms (GM12878-origin, adj. p < 0.05)
None of the significant terms matched cancer-relevant keywords (apoptosis, DNA damage/repair, p53, cell cycle, differentiation, proliferation, senescence, oncogene).

## Results — K562-origin switchers
628 terms tested across 3 libraries. 0 significant (adj. p < 0.05) shown below.

(none significant, or no results)

### Cancer-relevant terms (K562-origin, adj. p < 0.05)
None of the significant terms matched cancer-relevant keywords.

## Open question this was meant to address
Does the enrichment picture differ between GM12878-origin and K562-origin
AND<->INC switchers in a way consistent with the Day 44 directional
inconsistency (GM12878 marks-only ΔAUC +0.0552, K562 −0.1439)? Compare the
two term lists above directly — do not assume consistency or difference
without reading both tables.

## Honest caveat
If either or both result sets show zero significant terms, that is a real,
reportable outcome (gene lists this size, ~hundreds to low thousands, often
lack power for GO/KEGG enrichment after multiple-testing correction) — not
a failure of this script to fix.
