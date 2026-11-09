"""Render the screened, source-linked parametric/VMD research reading list.

Published reading reports live with the parametric approach. Bibliography entries retain only
titles/authors; the annotated review is an HTML research artifact, not a new
Markdown guide. Claims below are literature findings or explicitly proposals.
"""
import csv
import html
import json
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'approaches'/'parametric'/'report'/'literature_review_2026_11_04'
PAPERS=[]


def add(group,title,venue,year,url,authors,finding,application,limitation,*,priority=False,preprint=False,year_note='',evidence='Publisher/author abstract and publication metadata'):
    PAPERS.append(dict(group=group,title=title,venue=venue,year=year,url=url,authors=authors,
                       finding=finding,proposed_application=application,limitation=limitation,
                       priority=priority,status='Preprint; peer-reviewed publication not verified' if preprint else 'Published',
                       year_note=year_note,evidence=evidence))


def papers():
    g='A. Adaptive and learned mode decomposition'
    add(g,'Density-Based Adaptive Mode Decomposition','IEEE Transactions on Signal Processing',2026,'https://doi.org/10.1109/TSP.2026.3675233','Hao Jia; Chunyu Yang; Cesar F. Caiafa; Zhe Sun; Feng Duan; Jordi Solé-Casals',
        'Relates VMD optimization to density clustering; introduces automatic mode detection, direct extraction, and selective variational refinement.',
        'Compare automatic mode-count/center initialization with fixed K; initialize a learned decomposition from spectral clusters.',
        'Adaptivity and speech reconstruction do not establish corpus-trained decomposition or compact formula descriptions.',priority=True)
    add(g,'Short-time variational mode decomposition','Signal Processing',2026,'https://doi.org/10.1016/j.sigpro.2025.110203','Hao Jia et al.',
        'Extends bandwidth minimization to windowed time-frequency data; dynamic variant permits time-varying center frequencies.',
        'Use as the short-time and moving-frequency baseline; distinguish independent window VMD from its window-coupled objective.',
        'Reported application is EEG; it does not establish speech compression. Publisher also lists a corrigendum whose scope should be checked before reproduction.',priority=True,
        year_note='Journal issue January 2026; available online July 2025.')
    add(g,'TFD-Net: Towards Intelligent Time-Frequency Mode Decomposition with Practical Applications','AAAI',2026,'https://ojs.aaai.org/index.php/AAAI/article/view/39009','Pingping Pan; Yunjian Zhang; Jinyi Liu',
        'Combines a physically motivated time-frequency encoder with query-based mode extraction to learn separation of nonstationary components.',
        'A strong neural decomposition baseline and a source of mode-query and ridge-learning ideas.',
        'A learned mode extractor is not itself a compact mathematical expression generator or speech codec.',priority=True)
    add(g,'C-EMDNet: A Nonlinear Morphological Deep Framework for Robust Speech Enhancement','Sensors',2026,'https://www.mdpi.com/1424-8220/26/6/1917','Kais Khaldi; Sahar Almenwer; Afrah Alanazi; Inam Alanazi; Anis Mohamed',
        'Uses a U-Net to estimate masks over a CEEMDAN time-IMF representation.',
        'Useful control demonstrating that learning masks on adaptive modes differs from learning the decomposition itself.',
        'Its described CEEMDAN front end is a decomposition stage; the paper does not demonstrate gradients through CEEMDAN or parameter-vector coding.',evidence='Publisher methods sections 4.1–4.4 inspected')
    add(g,'Plug-and-play learnable EMD module for time-domain speech separation','Applied Acoustics',2024,'https://doi.org/10.1016/j.apacoust.2024.110210','Xiaokang Yang; Wenjun Ke; Jianguo Wei',
        'Constructs a learnable EMD-inspired module, using neural operations to replace conventional envelope computation.',
        'A direct speech-specific comparison for learned decomposition rather than only downstream processing of fixed modes.',
        'Speech separation is the objective, not low-description-length IMF fitting; full text was not available in this review.',priority=True)
    add(g,'Variational generalized nonlinear mode decomposition: Algorithm and applications','Mechanical Systems and Signal Processing',2024,'https://doi.org/10.1016/j.ymssp.2023.110913','Hongbing Wang; Shiqian Chen; Wanming Zhai',
        'Combines time-frequency fusion, ridge-based mode-type identification and variational recovery of chirp and dispersive components.',
        'Test whether speech transients need localized/dispersive components alongside oscillatory ones.',
        'Its mode-type discrimination is an engineered procedure, not unrestricted learned function synthesis.',priority=True)
    add(g,'Data-driven adaptive chirp mode decomposition with application to machine fault diagnosis under non-stationary conditions','Mechanical Systems and Signal Processing',2023,'https://doi.org/10.1016/j.ymssp.2022.109997','Hongbing Wang; Shiqian Chen; Wanming Zhai',
        'Builds instantaneous-frequency initialization and recursive extraction for nonstationary chirp components.',
        'Compare demodulation and IF initialization against fitting raw oscillating IMFs.',
        'Data-driven here means signal-adaptive estimation; it does not mean a model trained across a speech corpus.',priority=True,year_note='Journal issue 2023; DOI/online version 2022.')
    add(g,'Adaptive Variational Nonlinear Chirp Mode Decomposition','ICASSP',2022,'https://doi.org/10.1109/ICASSP43922.2022.9746147','Hao Liang; Xinghao Ding; Andreas Jakobsson; Xiaotong Tu; Yue Huang',
        'Bayesian adaptation estimates amplitude/frequency trajectories and the active dictionary, reducing bandwidth-parameter dependence.',
        'Investigate learned bandwidths, uncertainty, and automatic component activity instead of fixing eight equal-budget modes.',
        'Bayesian signal adaptation is distinct from neural amortization; compact coefficient coding still has to be added.',priority=True,
        evidence='Publisher abstract and author-hosted methodological excerpt')
    add(g,'Learnable Empirical Mode Decomposition based on Mathematical Morphology','SIAM Journal on Imaging Sciences',2022,'https://doi.org/10.1137/21M1417867','Santiago Velasco-Forero; Romain Pagès; Jesús Angulo',
        'Learns EMD-like envelope operations using mathematical morphology, including learned structuring functions and envelope combinations.',
        'A principled alternative decomposition module if differentiable VMD proves too restrictive.',
        'Demonstrations concern hyperspectral classification, not compact speech synthesis.',priority=True)
    add(g,'Nonlinear Chirp Mode Decomposition: A Variational Method','IEEE Transactions on Signal Processing',2017,'https://doi.org/10.1109/TSP.2017.2731300','Shiqian Chen et al.',
        'Uses demodulation to handle chirps whose bandwidth makes standard VMD assumptions restrictive.',
        'A mathematical foundation for replacing constant centers with trajectories.',
        'Trajectories are not automatically low-dimensional coefficient vectors; decomposition and compact fitting remain separate.')
    add(g,'Variational Mode Decomposition','IEEE Transactions on Signal Processing',2014,'https://doi.org/10.1109/TSP.2013.2288675','Konstantin Dragomiretskiy; Dominique Zosso',
        'Jointly extracts modes by minimizing their bandwidth after demodulation, subject to signal reconstruction.',
        'Keep its constrained objective as the reference when deciding what a learned VMD module should preserve.',
        'Small bandwidth is not a theorem of small polynomial degree or short symbolic description.')

    g='B. Numerical fitting and sparse continuous curves'
    add(g,'Variable Projection algorithms: Theoretical insights and a novel approach for problems with large residual','Automatica',2025,'https://doi.org/10.1016/j.automatica.2025.112300','Guangyong Chen; Peng Xue; Min Gan; Jing Chen; Wenzhong Guo; C. L. Philip Chen',
        'Studies local convergence of variable projection and proposes VPLR for separable nonlinear problems with appreciable residuals.',
        'Eliminate linear amplitude coefficients by least squares while optimizing only nonlinear frequencies/shape parameters.',
        'Applicable only where parameters enter conditionally linearly; it does not solve arbitrary symbolic topology search or guarantee global optimality.',priority=True,
        evidence='Publisher abstract and author-preprint formulation/Jacobian sections inspected')
    add(g,'Weighted Hermite Variable Projection Networks for Classifying Visually Evoked Potentials','IEEE Transactions on Neural Networks and Learning Systems',2025,'https://doi.org/10.1109/TNNLS.2024.3475271','Tamás Dózsa; Carl Böck; Jens Meier; Péter Kovács',
        'Introduces a differentiable variable-projection representation with trainable weighted Hermite functions.',
        'A concrete example of backpropagation through a fitted function representation; transfer the mechanism to oscillatory atoms.',
        'Hermite functions and EEG classification are its chosen setting; speech and unrestricted formulas need separate validation.',priority=True,
        year_note='Volume/issue July 2025; early-access DOI 2024.')
    add(g,'Differentiable Time-Varying Linear Prediction in the Context of End-to-End Analysis-by-Synthesis','Interspeech',2024,'https://www.isca-archive.org/interspeech_2024/yu24b_interspeech.html','Chin-Yun Yu; György Fazekas',
        'Makes sample-wise time-varying linear prediction differentiable and efficient in an analysis-by-synthesis voice system.',
        'A structured speech baseline and guidance for learning parameters through the actual resynthesis operation.',
        'Source-filter voice synthesis does not demonstrate compact representations of arbitrary VMD modes.',priority=True)
    add(g,'Sinusoidal Frequency Estimation by Gradient Descent','ICASSP',2023,'https://doi.org/10.1109/ICASSP49357.2023.10095188','Ben Hayes; Charalampos Saitis; György Fazekas',
        'Uses a complex-exponential surrogate and Wirtinger gradients to improve frequency optimization despite highly nonconvex sinusoidal losses.',
        'Test this against direct real-frequency least squares on known tones/chirps before rejecting a generated oscillatory expression.',
        'Its demonstrated sinusoidal setting does not guarantee recovery of general AM-FM or arbitrary nested expressions.',priority=True,
        year_note='ICASSP 2023; preprint first posted 2022.',evidence='Author-hosted introduction and surrogate/Jacobian sections inspected')
    add(g,'Sparsest Univariate Learning Models Under Lipschitz Constraint','IEEE Open Journal of Signal Processing',2022,'https://bigwww.epfl.ch/publications/aziznejad2202.pdf','Shayan Aziznejad; Thomas Debarre; Michael Unser',
        'Derives continuous piecewise-linear minimizers and sparsest solutions for stated one-dimensional regularized regression problems.',
        'Use an optimized sparse curve representation as a well-defined reference against unconstrained symbolic search.',
        'Optimality is for its specified variational problem; it is not global optimality among all waveform functions.',priority=True,
        evidence='Author-hosted abstract and variational formulations inspected')
    add(g,'Coupled Splines for Sparse Curve Fitting','IEEE Transactions on Image Processing',2022,'https://bigwww.epfl.ch/publications/jover2201.pdf','Icíar Lloréns Jover; Thomas Debarre; Shayan Aziznejad; Michael Unser',
        'Uses sparsity-promoting variational curve fitting and complementary spline components with controlled smoothness.',
        'Transfer adaptive knot allocation and mixed-smoothness modeling to IMF envelopes or residual phase.',
        'The paper addresses closed image contours; its theorems cannot simply be claimed for speech waveforms.',
        evidence='Author-hosted abstract and introduction inspected')
    add(g,'Learning Activation Functions in Deep (Spline) Neural Networks','IEEE Open Journal of Signal Processing',2020,'https://bigwww.epfl.ch/publications/bohra2003.html','Pakshal Bohra; Joaquim Campos; Harshit Gupta; Shayan Aziznejad; Michael Unser',
        'Learns free-form activation shapes using spline parameterizations and second-order total-variation regularization.',
        'An alternative to treating sine/SnakeBeta as the only activation choices; count learned knots and coefficients explicitly.',
        'Free-form activation learning still has a representation and regularization prior; it is not automatic minimal symbolic expression discovery.')
    add(g,'Separable nonlinear least squares: the variable projection method and its applications','Inverse Problems',2003,'https://doi.org/10.1088/0266-5611/19/2/201','Gene H. Golub; Victor Pereyra',
        'Reviews elimination of conditionally linear parameters in nonlinear least-squares fitting.',
        'The foundational explanation for separating amplitude fitting from oscillatory shape/frequency fitting.',
        'A numerical optimization reference, not evidence that a particular speech representation is sufficient.')

    g='C. Learning the function representation'
    add(g,'FINER++: Building a Family of Variable-periodic Functions for Activating Implicit Neural Representation','IEEE Transactions on Pattern Analysis and Machine Intelligence',2026,'https://doi.org/10.1109/TPAMI.2026.3722883','Hao Zhu; Zhen Liu; Qi Zhang; Jingde Fu; Weibing Deng; Zhan Ma; Yanwen Guo; Xun Cao',
        'Extends multiple activation backbones to variable-periodic functions to adjust supported frequencies and address capacity/convergence mismatch.',
        'Test per-IMF coordinate-function fitting with adaptive frequency support instead of assuming fixed sine or SnakeBeta is enough.',
        'Examples are largely visual INRs; parameters are network weights, not automatically a short analytic formula.',priority=True,
        year_note='TPAMI early access August 2026; preprint 2024. Venue/date checked against indexed journal record; methods checked against author preprint.')
    add(g,'KAN: Kolmogorov–Arnold Networks','ICLR',2025,'https://proceedings.iclr.cc/paper_files/paper/2025/hash/afaed89642ea100935e39d39a4da602c-Abstract-Conference.html','Ziming Liu; Yixuan Wang; Sachin Vaidya; Fabian Ruehle; James Halverson; Marin Soljačić; Thomas Y. Hou; Max Tegmark',
        'Replaces scalar edge weights with learnable univariate spline functions; supports pruning and interpretable function fitting.',
        'A learned-shape comparison, with later symbolic simplification assessed separately.',
        'For a scalar-time IMF, the multivariate representation theorem is not a compression guarantee; all spline parameters must be counted.',priority=True,
        year_note='ICLR 2025; preprint 2024.')
    add(g,'Kolmogorov-Arnold Networks Meet Science','Physical Review X',2025,'https://doi.org/10.1103/4t7t-v19l','Ziming Liu; Max Tegmark; Pingchuan Ma; Wojciech Matusik; Yixuan Wang',
        'Adds tools for extracting modular structure and symbolic formulas from learned KANs, including multiplication-aware structures.',
        'Compare learning a flexible mode curve followed by symbolic simplification with direct expression generation.',
        'Symbol extraction can require structural choices or hypotheses; audit the final formula fit rather than only the network fit.',priority=True,
        evidence='Publisher methodological excerpt; final venue/year checked against author institutional publication record')
    add(g,'FINER: Flexible Spectral-Bias Tuning in Implicit NEural Representation by Variable-periodic Activation Functions','CVPR',2024,'https://liuzhen0212.github.io/finer/','Zhen Liu; Hao Zhu; Qi Zhang; Jingde Fu; Weibing Deng; Zhan Ma; Yanwen Guo; Xun Cao',
        'Uses variable-periodic activations and initialization to adjust the frequencies a coordinate network can represent effectively.',
        'A targeted ablation for oscillatory IMF fitting and initialization sensitivity.',
        'Demonstrated visual-signal results need audio-specific error/size validation.')
    add(g,'WIRE: Wavelet Implicit Neural Representations','CVPR',2023,'https://openaccess.thecvf.com/content/CVPR2023/html/Saragadam_WIRE_Wavelet_Implicit_Neural_Representations_CVPR_2023_paper.html','Vishwanath Saragadam; Daniel LeJeune; Jasper Tan; Guha Balakrishnan; Ashok Veeraraghavan; Richard G. Baraniuk',
        'Uses complex Gabor-wavelet nonlinearities to combine spatial/time localization with oscillation.',
        'Test localized oscillatory representations for transient or nonstationary modes.',
        'Vision experiments and INR capacity do not prove compact speech parameterization.',priority=True)
    add(g,'Implicit Neural Representations with Periodic Activation Functions','NeurIPS',2020,'https://proceedings.neurips.cc/paper_files/paper/2020/file/53c04118df112c13a8c34b38343b9c10-Paper.pdf','Vincent Sitzmann; Julien N. P. Martel; Alexander W. Bergman; David B. Lindell; Gordon Wetzstein',
        'Introduces SIREN and an initialization designed for periodic coordinate networks; includes sound representation.',
        'Use properly initialized per-IMF function networks as a capacity/control fit, counting their actual weights.',
        'A fitted network can approximate a curve without being its smallest symbolic description.')
    add(g,'Neural Networks Fail to Learn Periodic Functions and How to Fix It','NeurIPS',2020,'https://papers.nips.cc/paper_files/paper/2020/hash/1160453108d3e537255e9f7b931f4e90-Abstract.html','Liu Ziyin; Tilman Hartwig; Masahito Ueda',
        'Introduces the Snake periodic inductive bias and studies periodic extrapolation.',
        'Motivates periodic activations as an ablation rather than assuming standard MLP activations handle oscillation.',
        'This is the Snake paper, not a general optimality result for SnakeBeta or a speech compression study.')

    g='D. Generating expressions and measuring their size'
    add(g,'(Exhaustive) symbolic regression and model selection by minimum description length','Philosophical Transactions of the Royal Society A',2026,'https://doi.org/10.1098/rsta.2024.0584','Harry Desmond',
        'Combines bounded exhaustive expression enumeration, equivalence simplification, numerical fitting and minimum-description-length scoring.',
        'Audit short-expression search gaps and charge both expression structure and coefficient precision.',
        'Enumeration is bounded by the declared grammar/size; numerical coefficient fitting is still a separate source of error.',priority=True,
        year_note='Published April 2026; author preprint July 2025; DOI contains 2024 but that is not the publication year.',
        evidence='Publisher method, enumeration and MDL sections inspected; journal date checked against indexed record')
    add(g,'RAG-SR: Retrieval-Augmented Generation for Neural Symbolic Regression','ICLR',2025,'https://proceedings.iclr.cc/paper_files/paper/2025/hash/19a8e70828c01059631f913442ae31e6-Abstract-Conference.html','Hengzhe Zhang; Qi Chen; Bing Xue; Wolfgang Banzhaf; Mengjie Zhang',
        'Uses online learned symbolic-tree generation, evolutionary feature construction and retrieval of expressions found during search.',
        'Retrieve successful IMF subexpressions and reuse discoveries instead of generating each formula from scratch.',
        'Its benchmark success is not speech validation; retrieval must preserve train/test isolation.',priority=True)
    add(g,'Symbolic regression via MDLformer-guided search: from minimizing prediction error to minimizing description length','ICLR',2025,'https://proceedings.iclr.cc/paper_files/paper/2025/hash/a402493de088886740b5939f666a6e56-Abstract-Conference.html','Zihan Yu; Jingtao Ding; Yong Li; Depeng Jin',
        'Learns a description-length-related search objective to guide equation discovery beyond raw prediction error.',
        'Investigate a learned cost-to-fit critic for guiding both expression search and, later, decomposition updates.',
        'Its predicted objective is not an actual measured speech bitstream length; validate against saved formulas and coefficients.',priority=True)
    add(g,'ParFam -- (Neural Guided) Symbolic Regression via Continuous Global Optimization','ICLR',2025,'https://proceedings.iclr.cc/paper_files/paper/2025/hash/ca98452d4e9ecbc18c40da2aa0da8b98-Abstract-Conference.html','Philipp Scholl; Katharina Bieker; Hillary Hauger; Gitta Kutyniok',
        'Transforms a symbolic-regression search into continuous optimization over a rich parametric construction, with optional neural guidance.',
        'A strong alternative structure-search baseline that separates topology failure from constant-optimization failure.',
        'Its admissible parametric construction remains a restriction; the title does not grant global optimality for our IMF problem.',priority=True)
    add(g,'Pareto-Optimal Fronts for Benchmarking Symbolic Regression Algorithms','ICML',2025,'https://proceedings.mlr.press/v267/fong25b.html','Kei Sen Fong; Mehul Motani',
        'Constructs exhaustive-reference accuracy/size fronts for benchmark datasets and compares numerical optimizers.',
        'Report distance to a bounded reference frontier on small known-answer problems rather than only comparing stochastic methods.',
        'Absolute-reference claims apply within its investigated search space, not every possible IMF expression.',priority=True)
    add(g,'Improving Monte Carlo Tree Search for Symbolic Regression','NeurIPS',2025,'https://proceedings.neurips.cc/paper_files/paper/2025/hash/3a28dd36daed2e3d66a3a143ab8b53cc-Abstract-Conference.html','Zhengyao Huang et al.',
        'Combines specialized tree-search allocation with mutation/crossover jumps between expressions.',
        'Give promising IMF formulas local structural refinement instead of relying only on independent sequence samples.',
        'Finite-time guarantees depend on stated assumptions; they are not unrestricted global guarantees for speech.',priority=True)
    add(g,'SymFormer: End-to-End Symbolic Regression Using Transformer-Based Architecture','IEEE Access',2024,'https://doi.org/10.1109/ACCESS.2024.3374649','Martin Vastl; Jonáš Kulhánek; Jiří Kubalík; Erik Derner; Robert Babuška',
        'Predicts expression tokens and numerical constants together, then refines the constants numerically.',
        'Learn coefficient initialization alongside structure to reduce nonlinear fitting failures.',
        'An additional symbolic-regression reference, not presented here as a flagship venue or verified speech result.')
    add(g,'Transformer-based Planning for Symbolic Regression','NeurIPS',2023,'https://proceedings.neurips.cc/paper_files/paper/2023/hash/8ffb4e3118280a66b192b6f06e0e2596-Abstract-Conference.html','Parshin Shojaee; Kazem Meidani; Amir Barati Farimani; Chandan K. Reddy',
        'Adds MCTS lookahead and fitting/complexity feedback to pretrained symbolic generation.',
        'Search with actual fitting outcomes rather than ranking generated formulas solely by token probability.',
        'Search compute must be included when comparing with direct numerical fitting.',priority=True)
    add(g,'Deep Generative Symbolic Regression with Monte-Carlo-Tree-Search','ICML',2023,'https://proceedings.mlr.press/v202/kamienny23a.html','Pierre-Alexandre Kamienny; Guillaume Lample; Sylvain Lamprier; Marco Virgolin',
        'Uses a pretrained conditional mutation model and online search experience to improve expression discovery.',
        'A particularly close precedent for learning to edit IMF-specific expressions using prior fitted candidates.',
        'Synthetic pretraining can mismatch real speech-mode structure.',priority=True,
        year_note='Use the ICML proceedings citation; an ICLR workshop version also circulated.')
    add(g,'End-to-end Symbolic Regression with Transformers','NeurIPS',2022,'https://papers.neurips.cc/paper_files/paper/2022/hash/42eb37cdbefd7abae0835f4b67548c39-Abstract-Conference.html','Pierre-Alexandre Kamienny; Stéphane d’Ascoli; Guillaume Lample; François Charton',
        'Generates complete expressions including constants from sampled numerical data.',
        'A model baseline for directly generating per-IMF formulas and useful constant guesses.',
        'Finite learned priors and fitting refinements do not establish globally minimal formulas.')
    add(g,'Neural Symbolic Regression that scales','ICML',2021,'https://proceedings.mlr.press/v139/biggio21a.html','Luca Biggio; Tommaso Bendinelli; Alexander Neitz; Aurélien Lucchi; Giambattista Parascandolo',
        'Pretrains on procedurally generated expressions and uses the predictions to guide equation recovery.',
        'The basic training design for a signal-to-expression proposer, with a speech-specific synthetic curriculum.',
        'Generalization from generated equations to real IMFs must be measured rather than presumed.')
    add(g,'Deep symbolic regression: Recovering mathematical expressions from data via risk-seeking policy gradients','ICLR',2021,'https://arxiv.org/abs/1912.04871','Brenden K. Petersen et al.',
        'Learns a distribution over expression trees using rewards from numerical fit quality.',
        'The foundational expression-generation objective; compare exploratory or risk-seeking updates with our pilot reward.',
        'It is a search heuristic and cannot certify the globally smallest description of arbitrary data.',year_note='ICLR 2021; first preprint 2019.')
    add(g,'AI Feynman 2.0: Pareto-optimal symbolic regression exploiting graph modularity','NeurIPS',2020,'https://proceedings.neurips.cc/paper/2020/hash/33a854e247155d590883b93bca53848a-Abstract.html','Silviu-Marian Udrescu; Andrew Tan; Jiahai Feng; Orisvaldo Neto; Tailin Wu; Max Tegmark',
        'Uses learned function properties and graph modularity to decompose expression discovery while retaining an accuracy/description frontier.',
        'Investigate compositional structure and repeated subexpressions when discovering compact mode formulas.',
        'Scientific-equation benchmarks and modularity priors do not guarantee compact formulas for speech IMFs.')

    g='E. Joint optimization and economical learned decompositions'
    add(g,'Modality-Agnostic Variational Compression of Implicit Neural Representations','ICML',2023,'https://proceedings.mlr.press/v202/schwarz23a.html','Jonathan Richard Schwarz; Jihoon Tack; Yee Whye Teh; Jaeho Lee; Jinwoo Shin',
        'Learns compact function modulations/subnetwork selection and explicitly optimizes a rate-distortion objective, including audio experiments.',
        'A later-stage reference for charging function-description costs and separating shared from signal-specific information.',
        'This is a shared-INR compression system; it does not answer our current explicit per-IMF formula-discovery question.',priority=True)
    add(g,'Fully learnable deep wavelet transform for unsupervised monitoring of high-frequency time series','PNAS',2022,'https://doi.org/10.1073/pnas.2106598119','Gabriel Michau; Gaetan Frusque; Olga Fink',
        'Learns wavelet filters and thresholds while incorporating structured filter-bank properties and sparse representations.',
        'A learned decomposition alternative and a model for testing constrained versus unconstrained filter learning.',
        'Monitoring and classification results do not demonstrate low-rate speech or symbolic mode descriptions.',priority=True)
    add(g,'Efficient and Modular Implicit Differentiation','NeurIPS',2022,'https://proceedings.neurips.cc/paper_files/paper/2022/hash/228b9279ecf9bbafe582406850c57115-Abstract-Conference.html','Mathieu Blondel et al.',
        'Differentiates solutions of optimization problems through their optimality conditions.',
        'Investigate differentiating a converged continuous fitter inside a learned decomposition objective.',
        'Regularity/local-solution assumptions matter; discrete expression changes cannot simply be differentiated as continuous variables.',priority=True)
    add(g,'LEAF: A Learnable Frontend for Audio Classification','ICLR',2021,'https://openreview.net/forum?id=jM76BCb6F9m','Neil Zeghidour; Olivier Teboul; Félix de Chaumont Quitry; Marco Tagliasacchi',
        'Learns structured audio filtering and subsequent feature operations with few frontend parameters.',
        'A constrained-filter initialization/control for learned bandwidth and center-frequency adaptation.',
        'Magnitude pooling/compression discard information; LEAF cannot be used as a reconstruction-preserving decomposition without changes.')
    add(g,'DDSP: Differentiable Digital Signal Processing','ICLR',2020,'https://openreview.net/forum?id=B1x1ma4tDr','Jesse Engel; Lamtharn Hantrakul; Chenjie Gu; Adam Roberts',
        'Integrates structured audio synthesis operations into neural training.',
        'A methodological precedent for supervising the actual waveform produced by fitted parameters.',
        'A fixed DSP synthesis structure is not unrestricted function generation or VMD.',priority=True)
    add(g,'Learning Sparse Neural Networks through L_0 Regularization','ICLR',2018,'https://openreview.net/forum?id=H1Y8hhg0b','Christos Louizos; Max Welling; Diederik P. Kingma',
        'Uses hard-concrete stochastic gates to learn exact zeros with differentiable expected sparsity penalties.',
        'Adapt the gating mechanism to active modes, knots or expression terms instead of allocating the same budget to all modes.',
        'Expected nonzero counts are a training surrogate; verify realized parameter and structure counts after selection.')

    g='F. Direct speech/audio parameter-coding precedents'
    add(g,'Audio coding via EMD','Digital Signal Processing',2020,'https://doi.org/10.1016/j.dsp.2020.102770','Abdel-Ouahab Boudraa; Kais Khaldi; Thierry Chonavel; Mounia Turki Hadj-Alouane; Ali Komaty',
        'Encodes local extrema of frame-wise EMD modes and incorporates a psychoacoustic model.',
        'A direct baseline for sparse mode descriptions and perceptually allocated detail.',
        'Extrema coding differs from generated analytic expressions and does not establish a learned VMD objective.',priority=True)
    add(g,'HHT-based audio coding','Signal, Image and Video Processing',2015,'https://doi.org/10.1007/s11760-013-0433-6','Kais Khaldi; Abdel-Ouahab Boudraa; Bruno Torrésani; Thierry Chonavel',
        'Codes Hilbert-derived instantaneous amplitude/frequency information from EMD modes, using compact parametric modeling.',
        'The close historical precedent for describing modes through their modulation trajectories.',
        'Does not establish our joint learned-VMD plus generated-expression setup; inspect its signal classes and rate accounting.',priority=True,
        year_note='Final volume/issue 2015; online first 2013. Author manuscript: https://hal.science/hal-00818033')
    add(g,'Perceptual Long-Term Variable-Rate Sinusoidal Modeling of Speech','IEEE Transactions on Audio, Speech, and Language Processing',2007,'https://www.gipsa-lab.grenoble-inp.fr/~laurent.girin/papers/LG_MF_SM_TASLP_2007.pdf','Laurent Girin; Mohammad Firouzmand; Sylvain Marchand',
        'Fits amplitude/phase trajectories of voiced-speech partials with perceptual criteria and variable-rate descriptions.',
        'A strong reference for fitting modulation trajectories and allocating error by perceptual relevance.',
        'Voiced sinusoidal partials and long sections differ from arbitrary 40 ms VMD modes.',priority=True,
        evidence='Author-hosted abstract and analysis/synthesis introduction inspected')

    g='G. Evaluation and survey entry points'
    add(g,'MAPSS: Manifold-based Assessment of Perceptual Source Separation','ICLR',2026,'https://proceedings.iclr.cc/paper_files/paper/2026/hash/8516e0109e6c4a3cdd17645dd61af7c7-Abstract-Conference.html','Amir Ivry; Samuele Cornell; Shinji Watanabe',
        'Separates perceptual leakage and self-distortion using learned representations and manifold-based measures.',
        'Consider for later full-speech evaluation; first verify applicability beyond source separation.',
        'High frame-rate outputs do not establish validity on an isolated 40 ms IMF; do not replace current fit diagnostics with it.')
    add(g,'Recent Advances in Symbolic Regression','ACM Computing Surveys',2025,'https://doi.org/10.1145/3735634','Junlan Dong; Jinghui Zhong',
        'Reviews symbolic-regression methods, algorithmic developments and evaluation issues.',
        'A broad reading map for extending the generator/search baseline.',
        'Survey-level evidence; verify individual technical claims against the original papers.')
    add(g,'A review of differentiable digital signal processing for music and speech synthesis','Frontiers in Signal Processing',2024,'https://doi.org/10.3389/frsip.2023.1284100','Ben Hayes; Jordie Shier; György Fazekas; Andrew McPherson; Charalampos Saitis',
        'Reviews differentiable audio synthesis, optimization pathologies and practical synthesis structures.',
        'A useful guide to choosing differentiable oscillatory and source-filter components.',
        'A survey, not an experimental demonstration of our architecture.',year_note='Journal publication 2024; DOI/preprint contain 2023.')

    g='H. Relevant preprints; not verified published papers'
    add(g,'Robust Spatiotemporal Forecasting Using Adaptive Deep-Unfolded Variational Mode Decomposition','arXiv preprint 2509.00703',2025,'https://arxiv.org/abs/2509.00703','Osama Ahmad; Lukas Wesemann; Fabian Waschkowski; Zubair Khalid',
        'Unrolls VMD updates and learns mode-specific bandwidth parameters and a complex multiplier; presents a two-stage traffic-forecasting pipeline.',
        'The closest found explicit neural-VMD implementation to inspect when designing a differentiable VMD block.',
        'ArXiv says under review; no peer-reviewed acceptance verified. Its objective and evaluation are not speech compression.',priority=True,preprint=True,
        evidence='Author-preprint method equations and two-stage architecture description inspected')
    add(g,'Symbolic Regression with Multimodal Large Language Models and Kolmogorov Arnold Networks','arXiv preprint 2505.07956',2025,'https://arxiv.org/abs/2505.07956','Thomas R. Harvey; Fabian Ruehle; Kit Fraser-Taliente; James Halverson',
        'Proposes expressions from function plots, numerically fits constants, evolves candidates and can symbolify learned KAN edge functions.',
        'A direct function-proposal alternative that may use a much broader implicit mathematical vocabulary.',
        'No publication acceptance verified; LLM priors, API costs and reproducibility require separate evaluation. It does not guarantee an unconstrained optimum.',preprint=True,
        evidence='Author-preprint proposal, numerical optimization and genetic-loop sections inspected')


SYNTHESIS=[
('Main finding',
 'The most useful direction is to learn a decomposition whose modes have cheap, accurately fitted function descriptions. Standard VMD minimizes demodulated bandwidth; that objective is not the same as minimizing the number of coefficients or expression tokens needed to reproduce each mode. This is a proposed connection between the decomposition and fitting literature, not a result demonstrated by the listed papers.'),
('Three meanings of data-driven',
 'Signal-adaptive numerical decomposition (VMD, DD-ACMD, DAMD), learning a downstream model on precomputed modes (e.g. mode masks), and a corpus-trained differentiable decomposition are distinct. Our intended second experiment needs the third, trained against the measured consequences of function fitting. A paper using VMD before an LSTM does not establish this.'),
('Oscillation structure versus polynomial degree',
 'A rapidly oscillating mode can be described more economically through an envelope and phase than by a polynomial of its waveform. Test direct expression discovery against factorized amplitude/phase expression discovery; do not force every real mode into one AM-FM template. Chirp-aware decomposition and localized oscillatory components are useful alternatives to constant-center VMD. Zero crossings alone do not specify the required polynomial degree or numerical conditioning.'),
('Continuous fitting before judging expression quality',
 'For a candidate sum of nonlinear atoms, fit conditionally linear coefficients by least squares and optimize only the nonlinear shape parameters when the structure permits. Compare variable projection, robust multi-start nonlinear fitting, and the complex sinusoidal surrogate on controlled signals. A poor local fit is not evidence that the generated function cannot represent the mode.'),
('Expression generation should include refinement',
 'RAG-SR, neural MCTS and TPSR suggest retaining successful expressions, editing subtrees and using numerical fitting feedback during search. This is better motivated than relying solely on independent samples from a small pretrained sequence model. SymFormer/E2E motivate learning constant initialization too. Keep equal-budget random and search baselines to establish that learning contributes.'),
('Learned shapes are not automatically symbolic formulas',
 'KAN, sparse splines, SIREN, WIRE and FINER++ offer flexible curve models. They answer a useful capacity question, but their parameters include all required weights, knots and coefficients. A learned activation is not itself a short analytic expression. Symbolification is an additional approximation whose error and description size must be measured.'),
('A practical joint experiment',
 'Start with fixed VMD and validated per-mode fits. Next compare an unfolded VMD module, a chirp-aware variant, and a learned filter-bank control. Use per-mode bandwidth/center parameters and learnable activity gates. A first outer objective can combine waveform error after formula evaluation, complete expression-description cost, and constraints/regularization preserving the intended narrow-band mode interpretation. Do not supervise only the sum of unfitted sampled modes.'),
('Discrete structures and differentiable training',
 'For a fixed expression structure, unrolled or implicit differentiation can propagate through continuous coefficient fitting subject to regularity and solver accuracy. Changing expression topology is discrete. Use alternating symbolic search and decomposition updates, or validate a learned cost-to-fit surrogate against actual search outcomes. Calling the whole mixed discrete/continuous system differentiable would be inaccurate without specifying this mechanism.'),
('Measure what must actually be stored',
 'Record floating coefficient count, expression-tree/DAG structure, knot positions, precision, any per-clip mode metadata, and the residual description. Shared model assets are separate. Parameters eliminated from numerical optimization are still parameters to store. A learned mode count should be penalized through realized description cost; equally splitting coefficients across all modes is only a baseline.'),
('Proposed ablations',
 'Decomposition: fixed VMD, learned-bandwidth VMD, chirp-aware decomposition, structured learned filter bank. Fitting: direct expression search, amplitude/phase expression search, sparse spline/KAN capacity control. Optimization: ordinary multi-start, variable projection where separable, oscillatory surrogate. Search: random, pretrained-only, online retrieval/mutation. Budget: adaptive versus equal mode allocation. Evaluate each change at matched expression-description budgets and comparable search time.'),
('Validation and limits',
 'Use known-answer functions with increasing modulation/transient complexity; measure topology recovery, constant-fitting success and final waveform errors separately. On speech, keep speakers disjoint for learning but allow all samples of a test IMF when fitting that IMF. Report mean/median, threshold success and the error/size frontier. Perceptual conclusions require later whole-speech evaluation and listening; separation metrics do not become valid for a 40 ms IMF merely because they produce frame-level outputs.'),
('What this review does not establish',
 'The reviewed sources provide components of the proposed system. I did not verify a published paper demonstrating the exact combination of 40 ms speech, a corpus-trained VMD-like decomposition, generated per-mode expressions and an objective charging the complete stored description. That is a scoped search finding, not a novelty claim or a proof that such a paper does not exist.'),
]


def main():
    papers();OUT.mkdir(parents=True,exist_ok=True)
    metadata=dict(review_date='2026-11-04',scope='Focused research review for 40 ms speech IMF fitting and learned VMD; not a formal exhaustive systematic review.',
                  search='Publisher proceedings and journals, author preprints and institutional manuscripts; backward/forward title searches; emphasis on 2024–2026 and essential older foundations.',
                  verification='Publication year follows the cited conference/journal issue or early-access publication, not the year embedded in a DOI or the first preprint. Primary sources support technical summaries; full proof audits and reproductions were not performed.',
                  evidence='Per-paper evidence field distinguishes abstract/metadata screening from methodological sections inspected.',
                  exclusions='Generic VMD+predictor application papers were not treated as learned decomposition. Unverified conference acceptances were not promoted from arXiv to a published venue.',
                  paper_count=len(PAPERS),published=sum(r['status']=='Published' for r in PAPERS),groups=dict(Counter(r['group'] for r in PAPERS)))
    (OUT/'papers.json').write_text(json.dumps(dict(metadata=metadata,papers=PAPERS,synthesis=SYNTHESIS),indent=2,ensure_ascii=False),encoding='utf-8')
    with (OUT/'papers.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(PAPERS[0]));writer.writeheader();writer.writerows(PAPERS)
    esc=html.escape
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Parametric fitting and learned VMD — literature review</title>
<style>body{font:16px/1.55 system-ui;max-width:1500px;margin:2rem auto;padding:0 1.4rem;color:#17212b}h1{line-height:1.2}a{color:#075e9b}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #d7dfe7;padding:.6rem;vertical-align:top}th{background:#edf3f8;text-align:left;position:sticky;top:0}.priority{background:#fff9e8}.note{background:#edf3f8;padding:1rem;border-radius:6px}details{margin:.8rem 0}code{overflow-wrap:anywhere}input{font:inherit;padding:.5rem;width:min(700px,90%)}@media print{input{display:none}th{position:static}details{break-inside:avoid}}</style>
<h1>Parametric curve fitting and data-driven VMD</h1><p>Research reading review • 4 November 2026</p>'''
    page+='<div class="note">'+esc(metadata['scope'])+'<br>'+esc(metadata['verification'])+'<br>'+esc(metadata['evidence'])+'</div>'
    page+='<p>'+str(len(PAPERS))+' screened papers; '+str(metadata['published'])+' published and '+str(len(PAPERS)-metadata['published'])+' explicitly labeled preprints. Highlighted rows are priority reading. Applications below are proposals for our experiment, not claims demonstrated in speech by the cited papers.</p>'
    page+='<h2>Research implications</h2>'+''.join('<h3>'+esc(title)+'</h3><p>'+esc(text)+'</p>' for title,text in SYNTHESIS)
    page+='<h2>Publication table</h2><p><input id="filter" placeholder="Filter by title, year, venue, method or proposed application" aria-label="Filter papers"></p>'
    for group in dict.fromkeys(r['group'] for r in PAPERS):
        page+='<h3>'+esc(group)+'</h3><div style="overflow:auto"><table><thead><tr><th>Paper title</th><th>Venue</th><th>Year</th><th>Finding and proposed use</th><th>Limits / evidence</th></tr></thead><tbody>'
        for r in (p for p in PAPERS if p['group']==group):
            page+='<tr class="paper '+('priority' if r['priority'] else '')+'"><td><a href="'+esc(r['url'],quote=True)+'">'+esc(r['title'])+'</a><br><small>'+esc(r['authors'])+'</small></td><td>'+esc(r['venue'])+'<br><small>'+esc(r['status'])+'</small></td><td>'+str(r['year'])+'</td><td><b>Paper:</b> '+esc(r['finding'])+'<br><b>Proposed transfer:</b> '+esc(r['proposed_application'])+'</td><td>'+esc(r['limitation'])+'<br><small>'+esc(r['year_note'])+'<br>'+esc(r['evidence'])+'</small></td></tr>'
        page+='</tbody></table></div>'
    page+='<p>Search coverage: '+esc(metadata['search'])+'</p><p>Exclusion rule: '+esc(metadata['exclusions'])+'</p>'
    page+='<script>document.getElementById("filter").addEventListener("input",e=>{const q=e.target.value.toLowerCase();for(const row of document.querySelectorAll("tr.paper"))row.hidden=!row.textContent.toLowerCase().includes(q)});</script></html>'
    (OUT/'index.html').write_text(page,encoding='utf-8')
    # Preserve the existing simple bibliography convention, including partial
    # author lists where the screened source only established the lead author.
    ledger=ROOT/'bibliography.md';existing=ledger.read_text(encoding='utf-8');added=[]
    for r in PAPERS:
        if r['title'].casefold() not in existing.casefold():added.append(r['title']+' — '+r['authors'])
    if added:
        with ledger.open('a',encoding='utf-8') as stream:stream.write('\n'+'\n'.join(added)+'\n')
    print(json.dumps(dict(**metadata,html=str(OUT/'index.html'),csv=str(OUT/'papers.csv'),bibliography_added=len(added)),indent=2))


if __name__=='__main__':main()
