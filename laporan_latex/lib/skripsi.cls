% To Do
% perbaiki jarak setelah figure dengan subsubsection
% perbaiki jarak setelah table dengan subsubsection
% buat header table center tapi isinya rata kiri

%===========================================================
% Skripsi LaTeX Class Style
%
% Original author:
% Dir Sulaiman
% https://github.com/dirsulaiman/skripsi
% 
% Thanks and reference to:
% Andi Kresna Jaya, S.Si.M.Si (unhasskripsi-latex)
% Guntur D Putra (https://github.com/gtrdp/template-skripsi)
% 
% License:
% BSD 3-Clause "New" or "Revised" License
% https://github.com/dirsulaiman/skripsi/blob/master/LICENSE
%-----------------------------------------------------------

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{lib/skripsi}[2020/03/11 Skripsi Class]
\LoadClass[a4paper, 12pt]{report}
\usepackage[margin=3cm, left=4cm]{geometry}

%===========================================================
% Require Package
%-----------------------------------------------------------
% to add indent for first paragraph
\RequirePackage{indentfirst}

% to justify
\RequirePackage{ragged2e}

% Prevent hyphenating the words
\RequirePackage[none]{hyphenat}
\sloppy

% Times New Roman Font like
\RequirePackage{newtxtext, newtxmath}

% to use or import image
\RequirePackage{graphicx}
\RequirePackage[export]{adjustbox}

% to use color in table
\RequirePackage[table]{xcolor}

% use CSV to generate table
\RequirePackage{csvsimple}

% to set space of paragraph line
\RequirePackage{setspace}
\onehalfspacing

% to customize Table of contents
\RequirePackage{tocloft}

% to customize footer and header
\RequirePackage{fancyhdr}

% to hide Hyperlink color
\RequirePackage[pdftex,bookmarks=true,hidelinks]{hyperref}

% Counter for figure, table and equation
\RequirePackage{chngcntr}

% Setup the caption of figure, table and equation
\RequirePackage[font=small, labelfont={},textfont={}, justification=centering]{caption}
\captionsetup{labelsep=space}

% Setup label of enumerate item
\RequirePackage{enumitem} 
% \renewcommand{\theenumi}{\Alph{enumi}}

% use Watermark
\RequirePackage[pages=some]{background}

% use listings for code script
\RequirePackage{listings}
\RequirePackage{color}

% float [H]
\RequirePackage{float}

% titlesec
\RequirePackage{titlesec}
\RequirePackage{apacite} 
\RequirePackage{longtable}

\RequirePackage{pdflscape}
\RequirePackage{dirtytalk}
\RequirePackage{lipsum}
\RequirePackage{inconsolata}
\RequirePackage{courier}
%-----------------------------------------------------------
% End Require Package
%===========================================================

\renewcommand{\baselinestretch}{1.5}
\renewcommand{\UrlFont}{\small\rm}

\newcommand{\listappendicesname}{DAFTAR LAMPIRAN}
\newlistof{appendices}{apc}{\listappendicesname}
\newcounter{appendixcounter}

\renewcommand{\cftappendicesleader}{\relax}
\renewcommand{\cftappendicesleader}{\hfill}

\newcommand{\newappendix}[1]{
    \refstepcounter{appendixcounter}
    \begin{center}
        \vspace{0.5em}
        {Lampiran \theappendixcounter\ #1}
    \end{center}
    \addcontentsline{apc}{appendices}{Lampiran \theappendixcounter\ #1}
}


%===========================================================
% Package Daftar Pustaka, bentuk citasi
% Config
% - style, citestyle = {authoryear, numeric, apa, mla}
% - sorting {year,name,title} = ynt
%-----------------------------------------------------------

\bibliographystyle{apacite}


% \usepackage[
%     backend=biber,
%     style=apa,
%     % citestyle=authoryear-comp,
%     % sorting=nty,
%     % giveninits=false,
%     % maxbibnames=3
%     ]{biblatex}
% \addbibresource{bibtex/daftar-pustaka.bib}

% redefine "et al" in bib to "dkk"
% \DefineBibliographyStrings{english}{andothers={dkk\adddot}}
% force to use quote in title field of bibliographi
% \DeclareFieldFormat
%   [article,inbook,incollection,inproceedings,patent,thesis,masterthesis,misc,unpublished, book]
%   {title}{\mkbibquote{#1\isdot}}


% new citation style
% \let\oldcitate\cite
% \newcommand*\thecite[1]{\hyperlink{cite.\therefsection @#1}{(\citeauthor*{#1}, \citeyear{#1})}}
% \renewcommand*\cite[1]{\hyperlink{cite.\therefsection @#1}{(\citeauthor*{#1}, \citeyear{#1})}}

\renewcommand{\bibname}{\begin{center}\textbf{\Large DAFTAR PUSTAKA}\end{center}}

%-----------------------------------------------------------
% End Biblatex Config
%===========================================================


%===========================================================
% Setting footer, header and page number
%-----------------------------------------------------------
\pagestyle{fancy}

\fancypagestyle{empty}{ %style for cover
    \fancyhf{} % Clear all headers and footers
    \renewcommand{\headrulewidth}{0pt} % Remove header line
    \renewcommand{\footrulewidth}{0pt} % Remove footer line
}
\fancypagestyle{roman}{ % Style for Roman numeral pages
    \fancyhf{} % Clear all headers and footers
    \fancyfoot[C]{\thepage} % Roman numeral page number centered at the bottom
    \renewcommand{\headrulewidth}{0pt} % Remove header line
    \renewcommand{\footrulewidth}{0pt} % Remove footer line
}

\fancypagestyle{fancy}{ % Style for chapter pages
    \fancyhf{} % Clear all headers and footers
    \fancyhead[R]{\thepage} % Page number at the top-right corner
    \renewcommand{\headrulewidth}{0pt} % Remove header line
    \renewcommand{\footrulewidth}{0pt} % Remove footer line
}

% Redefine "plain" style used by TOC and other sections
\fancypagestyle{plain}{
    \fancyhf{} % Clear all headers and footers
    \fancyfoot[C]{\thepage} % Roman numeral page number centered at the bottom
    \renewcommand{\headrulewidth}{0pt} % Remove header line
    \renewcommand{\footrulewidth}{0pt} % Remove footer line
}

% Redefine "plain" style for chapter pages
\fancypagestyle{chapterplain}{
    \fancyhf{} % Clear all headers and footers
    \fancyhead[R]{\thepage} % Page number at the top-right corner
    \renewcommand{\headrulewidth}{0pt} % Remove header line
    \renewcommand{\footrulewidth}{0pt} % Remove footer line
}

%-----------------------------------------------------------
% End Setting footer, header and page number
%===========================================================


%===========================================================
% Code Script, Listing
%-----------------------------------------------------------
\definecolor{light-gray}{gray}{0.95}
\definecolor{dkgreen}{rgb}{0,0.6,0}
\definecolor{gray}{rgb}{0.5,0.5,0.5}
\definecolor{mauve}{rgb}{0.58,0,0.82}

\lstdefinestyle{python}{
    language=Python,
    emptylines=1,
    backgroundcolor=\color{light-gray},
    rulecolor=\color{light-gray},
    xleftmargin=1cm,
    framexleftmargin=1cm,
    columns=flexible,
    basicstyle={\linespread{0.85}\small\ttfamily},
    breaklines=true,
    breakatwhitespace=true,
    showstringspaces=false,
    tabsize=2,
    moredelim=**[is][\color{red}]{@}{@},
    keywordstyle=\color{blue},
    commentstyle=\color{dkgreen},
    stringstyle=\color{mauve},
}

\lstdefinestyle{shell}{
    language=bash,
    emptylines=1,
    backgroundcolor=\color{black},
    rulecolor=\color{black},
    columns=flexible,
    basicstyle={\small\ttfamily\color{white}},
    breaklines=true,
    breakatwhitespace=true,
    showstringspaces=false,
    tabsize=2,
    moredelim=**[is][\color{red}]{@}{@},
    keywordstyle=\color{blue},
    commentstyle=\color{dkgreen},
    stringstyle=\color{mauve},
}

\lstdefinestyle{plain}{
    emptylines=1,
    backgroundcolor=\color{light-gray},
    rulecolor=\color{light-gray},
    columns=flexible,
    basicstyle={\footnotesize\ttfamily\color{black}},
    breaklines=true,
    breakatwhitespace=true,
    tabsize=2
}

\lstdefinelanguage{yaml}{
  keywords={true,false,null,yes,no},
  keywordstyle=\color{blue},
  comment=[l]{\#},
  commentstyle=\color{gray},
  basicstyle=\ttfamily\footnotesize,
  sensitive=true,
  morecomment=[l]{\#},
  morestring=[b]',
  morestring=[b]"
}

\lstdefinestyle{yamlstyle}{
    backgroundcolor=\color{light-gray},
    basicstyle=\ttfamily\footnotesize,
    emptylines=1,
    breaklines=true,
    captionpos=b,
    tabsize=2,
    language=yaml,
    morekeywords={true,false,null},
    keywordstyle=\color{blue},
    commentstyle=\color{gray},
    stringstyle=\color{orange}
}

\lstdefinestyle{plainbox}{
    language=Python,
    backgroundcolor=\color{white},
    frame=single,
    rulecolor=\color{gray},
    framesep=5pt,
    framerule=0.5pt,
    columns=fullflexible,
    basicstyle=\fontsize{10}{12}\selectfont\ttfamily,
    breaklines=true,
    breakatwhitespace=true,
    showstringspaces=false,
    tabsize=4
}
%-----------------------------------------------------------
% End Code Script, Listing
%===========================================================


%===========================================================
% Page Layout, Margin
%-----------------------------------------------------------
% \setlength{\topmargin}{0cm}
% \setlength{\headheight}{1cm}
% \setlength{\headsep}{0.2cm}
% \setlength{\topskip}{1ex}
% \setlength{\oddsidemargin}{1.2cm}
% \setlength{\evensidemargin}{1.2cm}
% \setlength{\textwidth}{14.3cm}
% \setlength{\textheight}{22.7cm}
% \setlength{\footskip}{1cm}
% \setlength{\marginparsep}{0pt}
% \setlength{\marginparwidth}{0pt}
%to set indent approx 6 character
\setlength{\parindent}{3em} 
%-----------------------------------------------------------
% End Page Layout, Margin
%===========================================================


% %===========================================================
% % Set Watermark
% %-----------------------------------------------------------
% \backgroundsetup{
%   scale=1,       
%   angle=0,       
%   opacity=.2,
%   hshift=3mm,
%   contents={\includegraphics[width=14cm]{images/logoUH.png}}
% }
% \newcommand\addwatermark{\BgThispage}
% %-----------------------------------------------------------
% % End Set Watermark
% %===========================================================


%===========================================================
% Redefition Terms
%-----------------------------------------------------------
\def\prefacename{KATA PENGANTAR}
\def\contentsname{DAFTAR ISI}
\def\listfigurename{DAFTAR GAMBAR}
\def\listtablename{DAFTAR TABEL}
\def\listappendixname{DAFTAR LAMPIRAN}
\def\nomenclaturename{DAFTAR LAMBANG DAN SINGKATAN}
\def\partname{BAGIAN}
\def\chaptername{BAB}
\def\appendixname{LAMPIRAN}
\def\refname{DAFTAR PUSTAKA}
\def\bibname{DAFTAR PUSTAKA}
\def\indexname{Indek}
\def\figurename{Gambar}
\def\tablename{Tabel}
\def\pagename{Halaman}
\renewcommand{\lstlistingname}{Potongan Kode}
\AtBeginDocument{
  \renewcommand{\thelstlisting}{\arabic{chapter}.\arabic{lstlisting}}
}
%-----------------------------------------------------------
% End Redefinition Terms
%===========================================================


%===========================================================
% Define thesis's inputs
%-----------------------------------------------------------
\newcommand{\titleskripsi}[1]{\gdef\@titleskripsi{#1}}
\newcommand{\@titleskripsi}{}
\newcommand{\titleeng}[1]{\gdef\@titleeng{#1}}
\newcommand{\@titleeng}{}
\newcommand{\idnum}[1]{\gdef\@idnum{#1}}
\newcommand{\@idnum}{}
\newcommand{\datesubmit}[1]{\gdef\@datesubmit{#1}}
\newcommand{\@datesubmit}{}
\newcommand{\monthsubmit}[1]{\gdef\@monthsubmit{#1}}
\newcommand{\@monthsubmit}{}
\newcommand{\yearsubmit}[1]{\gdef\@yearsubmit{#1}}
\newcommand{\@yearsubmit}{}
\newcommand{\@examdate}{\@datesubmit \space \@monthsubmit \space \@yearsubmit}
\newcommand{\degree}[1]{\gdef\@degree{#1}}
\newcommand{\@degree}{}
\newcommand{\program}[1]{\gdef\@program{#1}}
\newcommand{\@program}{}
\newcommand{\dept}[1]{\gdef\@dept{#1}}
\newcommand{\@dept}{}
\newcommand{\faculty}[1]{\gdef\@faculty{#1}}
\newcommand{\@faculty}{}
\newcommand{\university}[1]{\gdef\@university{#1}}
\newcommand{\@university}{}
\newcommand{\city}[1]{\gdef\@city{#1}}
\newcommand{\@city}{}
\newcommand{\firstsupervisor}[1]{\gdef\@firstsupervisor{#1}}
\newcommand{\@firstsupervisor}{}
\newcommand{\firstsupervisorNIP}[1]{\gdef\@firstsupervisorNIP{#1}}
\newcommand{\@firstsupervisorNIP}{}
\newcommand{\secondsupervisor}[1]{\gdef\@secondsupervisor{#1}}
\newcommand{\@secondsupervisor}{}
\newcommand{\secondsupervisorNIP}[1]{\gdef\@secondsupervisorNIP{#1}}
\newcommand{\@secondsupervisorNIP}{}
\newcommand{\firstexaminer}[1]{\gdef\@firstexaminer{#1}}
\newcommand{\@firstexaminer}{}
\newcommand{\firstexaminerNIP}[1]{\gdef\@firstexaminerNIP{#1}}
\newcommand{\@firstexaminerNIP}{}
\newcommand{\secondexaminer}[1]{\gdef\@secondexaminer{#1}}
\newcommand{\@secondexaminer}{}
\newcommand{\secondexaminerNIP}[1]{\gdef\@secondexaminerNIP{#1}}
\newcommand{\@secondexaminerNIP}{}
\newcommand{\headprogram}[1]{\gdef\@headprogram{#1}}
\newcommand{\@headprogram}{}
\newcommand{\headprogramNIP}[1]{\gdef\@headprogramNIP{#1}}
\newcommand{\@headprogramNIP}{}
\newcommand{\headdepartment}[1]{\gdef\@headdepartment{#1}}
\newcommand{\@headdepartment}{}
\newcommand{\headdepartmentNIP}[1]{\gdef\@headdepartmentNIP{#1}}
\newcommand{\@headdepartmentNIP}{}
\newcommand{\abstractID}[1]{\gdef\@abstractID{#1}}
\newcommand{\@abstractID}{}
\newcommand{\keywordsID}[1]{\gdef\@keywordsID{#1}}
\newcommand{\@keywordsID}{}
\newcommand{\abstractEN}[1]{\gdef\@abstractEN{#1}}
\newcommand{\@abstractEN}{}
\newcommand{\keywordsEN}[1]{\gdef\@keywordsEN{#1}}
\newcommand{\@keywordsEN}{}



% Init accumulators
\def\firstnames{}
\def\lastname{}
\def\@fullname{}

% Process fullname and split when set
\newcommand{\fullname}[1]{%
    \gdef\@fullname{#1}%
    \gdef\firstnames{}%
    \gdef\lastname{}%
    \splitname{#1}%
}

% Split logic
\def\splitname#1{%
    \begingroup
    \edef\@tempa{#1\space}%
    \expandafter\endgroup
    \expandafter\readwords\@tempa\relax
}

\def\readwords#1 #2\relax{%
    \ifx\relax#2\relax
        \gdef\lastname{#1}%
    \else
        \ifx\firstnames\empty
            \gdef\firstnames{#1}%
        \else
            \xdef\firstnames{\firstnames\space#1}%
        \fi
        \readwords#2\relax
    \fi
}

% Final output: now just prints, no processing
\newcommand{\lastnamefirstname}{%
    \lastname, \firstnames%
}


%-----------------------------------------------------------
% End Define thesis's inputs
%===========================================================


%===========================================================
% Define Cover Seminar Proposal/Hasil
%-----------------------------------------------------------
\def\coverproposal{
    \thispagestyle{empty}
    \pagenumbering{roman}
    \setcounter{page}{1}
    \begin{center}
            \MakeUppercase{\large\bfseries\@titleskripsi}\par\nobreak
            \vspace{1.5cm}
            \MakeUppercase{\large\bfseries{Proposal Skripsi}}\par\nobreak
            \vspace{1.5cm}
            \normalsize
            \textbf{Oleh:}\\
            \textbf{\@fullname \quad NIM.\@idnum}\\
            \vfill
            \includegraphics[height=1.56in]{images/polinema.png}
            \vfill
            
            \MakeUppercase{\large\bfseries Program Studi \@program}\\
            \MakeUppercase{\large\bfseries Jurusan \@dept}\\
            \MakeUppercase{\large\bfseries\@university}\\
            \MakeUppercase{\large\bfseries\@yearsubmit}\\
    \end{center}
    \pagebreak
}
%-----------------------------------------------------------
% End Define Cover Seminar Proposal/Hasil
%===========================================================


%===========================================================
% Define Cover Skripsi
%-----------------------------------------------------------
\def\cover{%
    \thispagestyle{empty}
    \pagenumbering{roman}
    \setcounter{page}{1}
    \begin{center}
            \MakeUppercase{\large\bfseries\@titleskripsi}\par\nobreak
            \vspace{1.5cm}
            \MakeUppercase{\large\bfseries{Skripsi}}\par\nobreak
            \normalfont{Digunakan Sebagai Syarat Maju Ujian Diploma IV Politeknik Negeri Malang}\\
            \vspace{1cm}
            \textbf{Oleh:}\\
            \textbf{\MakeUppercase{\@fullname} \quad NIM.\@idnum}\\
            \vfill
            \includegraphics[height=1.56in]{images/polinema.png}
            \vfill
            
            \MakeUppercase{\large\bfseries Program Studi \@program}\\
            \MakeUppercase{\large\bfseries Jurusan \@dept}\\
            \MakeUppercase{\large\bfseries\@university}\\
            \MakeUppercase{\large\bfseries{{\@monthsubmit} {\@yearsubmit}}}\\
    \end{center}
    \pagebreak
}
%-----------------------------------------------------------
% End Define Cover Skripsi
%===========================================================


%===========================================================
% Define Title Page
%-----------------------------------------------------------
\def\titlepage{
    \begin{center}
        \addwatermark
        \addcontentsline{toc}{chapter}{HALAMAN JUDUL}
        \begin{singlespace}
            \vspace*{2cm}
            \MakeUppercase{\normalfont\bfseries\@titleskripsi}\\
            \vfill

            \bfseries{SKRIPSI}
            \vfill

            \normalfont{Diajukan sebagai salah satu syarat untuk memperoleh gelar {\@degree} pada Program Studi {\@program} Departemen {\@dept} Fakultas {\@faculty} {\@university} {\@city}}\\
            \vfill

            {\normalfont\bfseries
                \@fullname\\
                \@idnum
            }\\
            \vfill

            {\normalfont
                \MakeUppercase{\normalfont\bfseries Program Studi \@program}\\
                \MakeUppercase{\normalfont\bfseries Departemen \@dept}\\
                \MakeUppercase{\normalfont\bfseries Fakultas \@faculty}\\
                \MakeUppercase{\normalfont\bfseries\@university}\\
                \MakeUppercase{\normalfont\bfseries\@city}\\
                \vspace{0.2cm}
                {\normalfont\bfseries{{\@monthsubmit} {\@yearsubmit}}}
                \vspace{2cm}
            }\\
        \end{singlespace}
    \end{center}
    \pagebreak
}
%-----------------------------------------------------------
% End Define Title Page
%===========================================================


%===========================================================
% Define Authentication Page
%-----------------------------------------------------------
\def\authenticationpage{
    \pagenumbering{roman}
    \setcounter{page}{3}
    \begin{center}
        \large
        \textbf{PERNYATAAN}\\
    \end{center}
    \vspace{0.5cm}
    \setlength{\parindent}{3em}
    \par
    Dengan ini saya menyatakan bahwa pada Skripsi ini tidak terdapat karya,
    baik seluruh maupun sebagian, yang sudah pernah diajukan untuk memperoleh
    gelar akademik di Perguruan Tinggi manapun, dan sepanjang pengetahuan saya
    juga tidak terdapat karya atau pendapat yang pernah ditulis atau diterbitkan oleh
    orang lain, kecuali yang secara tertulis disitasi dalam naskah ini serta disebutkan
    dalam daftar sitasi/pustaka.
    \vspace{2cm}
    \begin{flushright}
        Malang, \@datesubmit \\
        \vspace{2cm}
        % make fullname uppercase first letter words
        \@fullname \\
    \end{flushright}
    \pagebreak
}
%-----------------------------------------------------------
% End Define Authentication Page
%===========================================================

%===========================================================
% Define Absctract Page
%-----------------------------------------------------------
\def\abstractpageID{%
    \setlength{\parskip}{1em}
    \addcontentsline{toc}{chapter}{ABSTRAK}%
    \begin{center}
        \MakeUppercase{\normalfont\large\bfseries ABSTRAK}%
    \end{center}
    
    \noindent{\bfseries \lastnamefirstname}. \say{\@titleskripsi}.
    \textbf{Pembimbing:} (1) \@firstsupervisor, (2) \@secondsupervisor \par

    \noindent\textbf{Skripsi, Program Studi \@program, Jurusan \@dept, \@university, \@yearsubmit.}\par

    \@abstractID\par

    \noindent\textbf{Kata Kunci:} \@keywordsID

    \pagebreak
}


\def\abstractpageEN{%
    \setlength{\parskip}{1em}%
    
    \addcontentsline{toc}{chapter}{\textit{ABSTRACT}}%
    \begin{center}
        \MakeUppercase{\normalfont\large\bfseries\textit{ABSTRACT}}%
    \end{center}
    
    \noindent\textit{{\bfseries \lastnamefirstname}. \say{\@titleskripsi}.}
    \textit{\textbf{Supervisor:} \@firstsupervisor, \textbf{Co-Supervisor:} \@secondsupervisor}\par

    \noindent\textit{\textbf{Thesis, Information Engineering Study Program, Department of Information Technology, State Polytechnic of Malang, \@yearsubmit.}}\par

    \textit{\@abstractEN}\par

    \noindent\textit{\noindent\textbf{Keywords:} \@keywordsEN}

    \pagebreak
}

%-----------------------------------------------------------

%===========================================================
% Define Approval Page
%-----------------------------------------------------------
\def\approvalpage{%
    \thispagestyle{roman}
    \begin{center}
        \large
        \textbf{HALAMAN PENGESAHAN}\\
        \textbf\uppercase{\@titleskripsi}\\
        \vspace{1cm}
        \normalsize
        \textbf{Disusun oleh:}\\
        \textbf{\@fullname \quad NIM. \@idnum}\\
        \vspace{1cm}
        \textbf{Laporan Akhir ini telah diuji pada tanggal}\\
        \textbf{Disetujui oleh:}

        \vfill
        \begin{tabular}{p{0.2cm}lp{7.2cm}l}
            \parbox{0.5cm}{1. \\} & \parbox{2.0cm}{Pembimbing Utama\\} &\parbox{0.5cm}{:\\}\parbox{6cm}{\underline{\@firstsupervisor}\\NIP. \@firstsupervisorNIP} & \parbox{2.0cm}{........................}\\ \\
            \parbox{0.5cm}{1. \\} & \parbox{2.0cm}{Pembimbing Pendamping\\} &\parbox{0.5cm}{:\\}\parbox{6cm}{\underline{\@secondsupervisor}\\NIP. \@secondsupervisorNIP} & \parbox{2.0cm}{........................}\\ \\
            \parbox{0.5cm}{2. \\} & \parbox{2.0cm}{Penguji Utama \\} &\parbox{0.5cm}{:\\}\parbox{6cm}{\underline{\@firstexaminer}\\NIP. \@firstexaminerNIP} & \parbox{2.0cm}{........................}\\ \\                        \parbox{0.5cm}{3. \\} & \parbox{2.1cm}{Penguji Pendamping\\} &\parbox{0.5cm}{:\\}\parbox{6cm}{\underline{\@secondexaminer}\\NIP. \@secondexaminerNIP} & \parbox{2.0cm}{}{........................}\\ \\
        \end{tabular}
        \vfill
        Mengetahui,
        \vfill
        \centering
        \begin{tabular}{ >{\centering\arraybackslash}m{7.2cm}  m{0cm} >{\centering\arraybackslash}m{6cm} }
       	    Ketua Jurusan & & Ketua Program Studi \\
        	Teknologi Informasi & & Teknik Informatika \\
            & & \\
            & & \\
            & & \\
            & & \\
            \underline{\@headdepartment} & & \underline{\@headprogram} \\
            NIP. \@headdepartmentNIP & & NIP. \@headprogramNIP
        \end{tabular}
    \end{center}
    \pagebreak
}
%-----------------------------------------------------------
% End Define Approval Page
%===========================================================


%===========================================================
% Define Persetujuan Publikasi Karya Ilmiah
%-----------------------------------------------------------
\def\publicationapprovalpage{%
    % \addwatermark
    \addcontentsline{toc}{chapter}{PERSETUJUAN PUBLIKASI KARYA ILMIAH}
    \begin{center}
        \begin{singlespace}
            \MakeUppercase{\normalfont\large\bfseries\expandafter{PERNYATAAN PERSETUJUAN PUBLIKASI TUGAS AKHIR UNTUK KEPENTINGAN AKADEMIS}}\\
            \vspace{1.0cm}
        \end{singlespace}
    \end{center}

    \begin{singlespace}
        \noindent
        Sebagai civitas akademik {\@university}, saya yang bertanda tangan di bawah ini:
        \begin{center}
            \begin{tabular}{ p{3cm}  p{.1cm} p{10.5cm}}
                Nama &:& \@fullname\\
                NIM &:& \@idnum\\
                Program Studi &:& \@program\\
                Departemen &:& \@dept\\
                Fakultas &:& \@faculty\\
                Jenis Karya &:& Skripsi
            \end{tabular}
        \end{center}
    \end{singlespace}

    \noindent
    Demi pengembangan ilmu pengetahuan, menyetujui untuk memberikan kepada {\@university} \textbf{Hak Predikator Royalti Noneksklusif} (\textit{Non-exclusive Royalty-Free Right}) atas tugas akhir saya yang berjudul:

    \begin{center}
        {\normalfont\bfseries\expandafter{"\@titleskripsi"}}\\
            % \vspace{1.0cm}
    \end{center}

    \noindent
    beserta perangkat yang ada (jika diperlukan). Terkait dengan hal diatas, maka pihak {\@university} berhak menyimpan, mengalih-media/format-kan, mengola dalam bentuk pangkalan data (\textit{database}), merawat dan mempublikasikan tugas akhir saya selama tetap mencantumkan nama saya sebagai penulis/pencipta dan sebagai pemilik Hak Cipta.

    \noindent
    Demikian surat pernyataan ini saya buat dengan sebenarnya.

    \vspace{.5cm}
    \begin{flushright}
        Dibuat di {\@city} pada \@examdate\\
        Yang menyatakan\\
        \vspace{2.5cm}
        {(\@fullname)}\\
    \end{flushright}
    \pagebreak
}

% kata pengantar/preface
\def\prefacepage{
    \addcontentsline{toc}{chapter}{KATA PENGANTAR}
    \begin{center}
        \MakeUppercase{\normalfont\large\bfseries KATA PENGANTAR}
    \end{center}
    \vspace{0.5cm}
    \setlength{\parskip}{0pt}
    \setlength{\parindent}{3em} \par
Puji syukur kami panjatkan ke hadirat Allah SWT/Tuhan YME atas segala rahmat dan hidayah-Nya sehingga penulis dapat menyelesaikan skripsi dengan judul {``\MakeUppercase{\textit{\@titleskripsi}}''}. Skripsi ini disusun sebagai salah satu syarat untuk menyelesaikan studi program Diploma IV Program Studi Teknik Informatika, Jurusan Teknologi Informasi, Politeknik Negeri Malang.

Penulis menyadari bahwa tanpa adanya dukungan, bimbingan, dan kerja sama dari berbagai pihak, penyusunan laporan akhir ini tidak akan berjalan dengan baik. Oleh karena itu, dengan segala kerendahan hati, penulis ingin menyampaikan rasa terima kasih kepada:

\begin{enumerate}
    \item Bapak \@headdepartment, selaku Ketua Jurusan \@dept.
    \item Ibu \@headprogram, selaku Ketua Program Studi D-IV \@program.
    \item Ibu \@firstsupervisor, selaku Dosen Pembimbing Utama yang telah memberikan bimbingan, arahan, dan motivasi kepada penulis.
    \item Ibu \@secondsupervisor, selaku Dosen Pembimbing Pendamping yang telah banyak membantu dalam proses penyusunan skripsi ini.
    \item Bapak Indrazno Siradjuddin, S.T., M.T., Ph.D., selaku Kepala Laboratorium Robotika yang telah memberikan fasilitas dan tempat untuk melakukan riset serta turut membantu dalam pelaksanaan penelitian ini.
    \item Seluruh tim robotika yang telah memberikan dukungan, bantuan, serta masukan yang sangat berharga selama proses penelitian berlangsung.
    \item Dan seluruh pihak yang telah membantu dan mendukung kelancaran pembuatan laporan akhir ini dari awal hingga selesai, yang tidak dapat disebutkan satu per satu.
\end{enumerate}

Penulis menyadari bahwa dalam penyusunan laporan ini masih terdapat banyak kekurangan, baik dalam penyampaian materi, sistematika penulisan, maupun penggunaan bahasa. Oleh karena itu, penulis mengharapkan kritik dan saran yang bersifat membangun demi perbaikan di masa mendatang.

Akhir kata, penulis berharap laporan ini dapat bermanfaat bagi pembaca pada umumnya dan bagi penulis khususnya. Semoga karya ini menjadi awal yang baik dalam penerapan ilmu yang telah diperoleh selama masa studi.

    \vspace{2cm}
    \begin{flushright}
    Malang, \@datesubmit \\
    \vspace{1.5cm}
    Penulis \\
    \end{flushright}
    \pagebreak
}
%-----------------------------------------------------------
% End Define Persetujuan Publikasi Karya Ilmiah
%===========================================================

\def\lampiran{
    \addcontentsline{toc}{chapter}{LAMPIRAN}
    \begin{center}
        \MakeUppercase{\normalfont\large\bfseries LAMPIRAN}
    \end{center}
    \setlength{\parskip}{0pt}
    \setlength{\parindent}{3em} \par
    \input{include/lampiran.tex}
}

%===========================================================
% Redefine chapter and sections
%-----------------------------------------------------------

\renewcommand \thechapter {\@Roman\c@chapter}
\renewcommand \thesection {\@arabic\c@chapter.\@arabic\c@section}
\renewcommand \thesubsection {\@arabic\c@chapter.\@arabic\c@section.\@arabic\c@subsection}
\renewcommand \thesubsubsection {\@arabic\c@chapter.\@arabic\c@section.\@arabic\c@subsection.\@arabic\c@subsubsection}

\renewcommand{\@makechapterhead}[1]{%
    {\parindent \z@ \centering \normalfont
    \ifnum \c@secnumdepth >\m@ne
        \large\bfseries \@chapapp\space \thechapter.{\space}\p@
    \fi
        \interlinepenalty\@M
        \large \bfseries #1\par\nobreak
        \vskip 15\p@
    }
}
\renewcommand{\section}{
    \@startsection {section}{1}{\z@}
        {12pt}
        {.2ex}
        {\normalfont\normalsize\bfseries}
}
\renewcommand{\subsection}{
    \@startsection{subsection}{2}{\z@}
        {12pt}
        {.2ex}
        {\normalfont\normalsize\bfseries}
}
\renewcommand{\subsubsection}{
    \@startsection{subsubsection}{2}{\z@}
        {-3.25ex\@plus -1ex \@minus -.2ex}
        {1.5ex \@plus .2ex}
        {\normalfont\normalsize\bfseries}
}

%-----------------------------------------------------------
% End Redefine chapter and sections
%===========================================================


%===========================================================
% Setting Tabble of Content, list of figure, list of table
%-----------------------------------------------------------
% customize Table of contents
\renewcommand{\cftchappresnum}{BAB }
\renewcommand{\cftchapnumwidth}{3.7em}
\renewcommand{\cftchapaftersnum}{.}
\renewcommand{\cftchapdotsep}{\cftnodots} % Removes dots from chapter titles
\renewcommand{\cftsecdotsep}{\cftnodots}  % Removes dots from sections
\renewcommand{\cftsubsecdotsep}{\cftnodots} % Removes dots from subsections
\renewcommand\cftchapfont{}
\renewcommand\cftchappagefont{\normalfont}
\setlength{\cftbeforechapskip}{3pt}   % Space before chapter entries
\setlength{\cftbeforesecskip}{3pt}   % Space before section entries
\setlength{\cftbeforesubsecskip}{3pt} % Space before subsection entries

% table of contents
\renewcommand{\cftbeforetoctitleskip}{-0.25cm}
\renewcommand{\cftaftertoctitleskip}{0.75cm}
\renewcommand\cfttoctitlefont{\hfill\large\bfseries\MakeUppercase}
\renewcommand\cftaftertoctitle{\hfill\mbox{}}
% list of figures
\renewcommand{\cftbeforeloftitleskip}{-0.25cm}
\renewcommand{\cftafterloftitleskip}{0.75cm}
\renewcommand\cftloftitlefont{\hfill\large\bfseries\MakeUppercase}
\renewcommand\cftafterloftitle{\hfill\mbox{}}
\renewcommand{\cftfigpresnum}{Gambar~}
\setlength{\cftfignumwidth}{2.5cm}
% list of tables
\renewcommand{\cftbeforelottitleskip}{-0.25cm}
\renewcommand{\cftafterlottitleskip}{0.75cm}
\renewcommand\cftlottitlefont{\hfill\large\bfseries\MakeUppercase}
\renewcommand\cftafterlottitle{\hfill\mbox{}}
\renewcommand{\cfttabpresnum}{Tabel~}
\setlength{\cfttabnumwidth}{2.5cm}

% list of appendix
\renewcommand{\cftbeforeapctitleskip}{-0.25cm}
\renewcommand{\cftafterapctitleskip}{0.75cm}
\renewcommand\cftapctitlefont{\hfill\large\bfseries\MakeUppercase}
\renewcommand\cftafterapctitle{\hfill\mbox{}}

\renewcommand{\cftchapdotsep}{\cftnodots} % No dots for chapter entries in ToC
\renewcommand{\cftsecdotsep}{\cftnodots}  % No dots for section entries in ToC
\renewcommand{\cftsubsecdotsep}{\cftnodots} % No dots for subsection entries in ToC
\renewcommand{\cftfigdotsep}{\cftnodots}  % No dots for figures in LoF
\renewcommand{\cfttabdotsep}{\cftnodots}  % No dots for tables in LoT

%-----------------------------------------------------------
% End Setting Tabble of Content, list of figure, list of table
%===========================================================


%===========================================================
% Set Counter for figure, table, and equation
%-----------------------------------------------------------
\counterwithin{figure}{chapter}
\counterwithin{table}{chapter}
\counterwithin{equation}{chapter}
\renewcommand \thefigure {\@arabic\c@chapter.\@arabic\c@figure}
\renewcommand \thetable {\@arabic\c@chapter.\@arabic\c@table}
\renewcommand \theequation {\@arabic\c@chapter.\@arabic\c@equation}
%-----------------------------------------------------------
% End Set Counter for figure, table, and equation
%===========================================================

%===========================================================
% Set Table, Figure Environment
%-----------------------------------------------------------
% new environment for Figure
\newenvironment{afigure}
    {\begin{figure}[ht]}
    {\end{figure}}

% new environment for Table
\newenvironment{atable}
    {\begin{table}[ht] \renewcommand{\arraystretch}{1.3} \centering \small}
    {\end{table}}

\newcommand*{\captionsource}[2]{%
    \caption[{#1}]{%
      #1%
      \\
      {Sumber:} #2%
    }%
  }
%-----------------------------------------------------------
% End Set Table, Figure Environment
%===========================================================


%===========================================================
% Set nosep for itemize and enumerate
%-----------------------------------------------------------
\renewcommand\labelitemi{\large$\bullet$}
\setlist[itemize]{noitemsep, topsep=0pt}
\setlist[enumerate]{noitemsep, topsep=0pt}
%-----------------------------------------------------------
% End Set nosep for itemize and enumerate
%===========================================================