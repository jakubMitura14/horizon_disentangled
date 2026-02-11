\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{nhrproposal}[Class for NHR detailed descriptions]

\RequirePackage{graphicx}
\RequirePackage{amsmath}
\RequirePackage{amssymb}
\RequirePackage[table]{xcolor}
\RequirePackage[T1]{fontenc}

\newcommand{\templateversion}{ 9. October 2025}

\DeclareOption{new}{
  \newcommand{\doctitle}{Detailed Description for New NHR Projects}
  \newcommand{\newproposal}{}
}
\DeclareOption{ext}{
  \newcommand{\doctitle}{Detailed Description for Extensions of NHR Projects}
  \newcommand{\extensionproposal}{}
}
\DeclareOption{whitelisted}{
  \newcommand{\doctitle}{Detailed Description for Already Reviewed NHR Normal Projects}
  \newcommand{\whitelistedproposal}{}
  }
\DeclareOption{whitelistedextension}{
  \newcommand{\doctitle}{Detailed Description for Extension of Already Reviewed NHR Normal Projects}
  \newcommand{\whitelistedextensionproposal}{}
  }
\DeclareOption{showhints}{
  \newcommand\showhints{}
}

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax

\LoadClass[11pt,a4paper]{article}
\RequirePackage[numbers]{natbib}
\RequirePackage{hyperref}
\RequirePackage{titlesec}
\RequirePackage{graphicx}
\RequirePackage{lmodern}
\RequirePackage{hyperref}
\RequirePackage{verbatim}
\RequirePackage{tabularx}
\RequirePackage{array}
\RequirePackage{xstring}
\RequirePackage{environ}
\RequirePackage{listofitems}
\RequirePackage{forloop}
\RequirePackage{fp}
\RequirePackage{xfp}
\RequirePackage[T1]{fontenc}
\RequirePackage{mathptmx}
\RequirePackage[a4paper,margin=2.5cm]{geometry}
\RequirePackage[headings]{fancyhdr}

\title{\doctitle}
\author{\projecttitle}
\date{\authors}
\date{\authors \expandafter\ifx\csname showhints\endcsname\relax\else
\begingroup
\\
\vspace{1cm}
Version \templateversion
\endgroup\fi}

\definecolor{nhr}{rgb}{0.988, 0.122, 0.678}
\definecolor{NHR}{rgb}{0.988, 0.122, 0.678}
\definecolor{Gray}{gray}{0.85}

\titleformat{\section}{\normalfont\normalsize\bfseries}{\thesection}{1em}{}
\titleformat{\subsection}{\normalfont\normalsize\bfseries}{\thesubsection}{1em}{}
\titleformat{\subsubsection}{\normalfont\normalsize}{\thesubsubsection}{1em}{}
\titleformat{\paragraph}{\normalfont\normalsize}{\theparagraph}{1em}{}



\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}
\newcolumntype{G}[1]{>{\columncolor{Gray}\raggedright\arraybackslash}p{#1}}
\newcolumntype{C}[1]{>{\centering\arraybackslash}p{#1}}
\newcolumntype{R}[1]{>{\raggedleft\arraybackslash}p{#1}}

\pagenumbering{arabic}
\setlength\parindent{0pt}

\newcommand{\reslabel}[1]{
  \IfSubStr{#1}{CPU}
  {
    [mio. CPU-core-hours]
  }
  {
    [thousand #1-hours]
  }
}
\def\hlinewd#1{%
\noalign{\ifnum0=`}\fi\hrule \@height #1 \futurelet
\reserved@a\@xhline}

\renewcommand{\theparagraph}{\arabic{paragraph}.}
\setcounter{secnumdepth}{4}

\newcommand{\Titlepage}{
  \thispagestyle{empty}
  \sffamily
  \makeatletter
    \begin{titlepage}
    \expandafter\ifx\csname projectlogo\endcsname\relax
        \begin{center}
            \includegraphics[width=0.3\linewidth]{figures/NHR_Logo_RGB_farbig_vertikal.png}\\[4ex]
        \end{center}
      \else
            \includegraphics[width=0.3\linewidth]{figures/NHR_Logo_RGB_farbig_vertikal.png}
        \hfill
            \includegraphics[width=0.3\linewidth]{\projectlogo}\\[4ex]
      \fi
        \begin{center}
            {\Huge \bfseries \@title }\\[10ex]
            {\huge  \@author}\\[10ex]
            {\huge  \@date}
        \end{center}
    \end{titlepage}
  \makeatother
  \hypersetup{
    pdftitle={\doctitle: \ \projecttitle},
    pdfsubject={\projecttitle},
    pdfauthor={\authors}
  }
  \setcounter{page}{1}
  \pagestyle{plain}
  \fancyhf{}
  \renewcommand{\headrulewidth}{0pt}
  \fancyhead[R]{\leftmark}
  \fancyfoot[C]{\thepage}

  \expandafter\ifx\csname showhints\endcsname\relax\else
\begingroup
\itshape
{\color{nhr}{Please note the following and remove this page in final version by removing the option \textbf{showhints} from the documentclass.}}
\vspace{0.5cm}

\expandafter\ifx\csname newproposal\endcsname\relax\else
\textbf{The length of the project description should not exceed 12 pages (font 11 pt, not including the references)!}
\vspace{0.5cm}

Note: Fill-out hints are given in italic. \textbf{Please remove them in the final version by removing the option \textbf{showhints} from the documentclass.}

\vspace{1cm}

Before using this template, please check at \url{https://go-nhr.de/computing-time} if a new version is available (this is version \templateversion).

\vspace{1cm}

When to use this template:
\begin{itemize}
  \item for \textbf{\underline{new} NHR normal and NHR large projects}  
\end{itemize}

When \textbf{not} to use this template:  
\begin{itemize}
\item Don't use this template for extensions of NHR normal and NHR large projects. For extensions of projects, please use the simplified template available at \url{https://go-nhr.de/computing-time}. 
\item \textbf{Already reviewed research projects:} If you are applying for an NHR normal project and if your scientific project has already been successfully reviewed and approved by one of the accepted institutions (DFG, European Union, German Federal Ministries BM* (e.g. BMFTR; BMWK), Volkswagen Stiftung, another NHR or GCS center) depending on the NHR center you can use a simplified template available at \url{https://go-nhr.de/computing-time}. 
\end{itemize}
\fi






\expandafter\ifx\csname extensionproposal\endcsname\relax\else
\textbf{The length of the project description should not exceed 14 pages (font 11 pt, not including the references)!}
\vspace{0.5cm}

Note: Fill-out hints are given in italic. \textbf{Please remove them in the final version by removing the option \textbf{showhints} from the documentclass.}

\vspace{1cm}

Before using this template, please check at \url{https://go-nhr.de/computing-time} if a new version is available (this is version \templateversion).

\vspace{1cm}

When to use this template:
\begin{itemize}
  \item for \textbf{\underline{extensions} of NHR normal and NHR large projects}  
\end{itemize}
When \textbf{not} to use this template:  
\begin{itemize}
  \item Don't use this template for \underline{\textbf{short-term}} extensions (up to three months). Contact your NHR center instead. 
  \item Don't use this template for \textbf{new} NHR normal and NHR large projects. For new projects, please use the template available at \url{https://go-nhr.de/computing-time}. 
\item \textbf{Already reviewed research projects:} If you are applying for an NHR normal project and if your scientific project has already been successfully reviewed and approved by one of the accepted institutions (DFG, European Union, German Federal Ministries BM* (e.g. BMFTR; BMWK), Volkswagen Stiftung, another NHR or GCS center) depending on the NHR center you can use a simplified template available at \url{https://go-nhr.de/computing-time}. 
\end{itemize}
  \fi






\expandafter\ifx\csname whitelistedproposal\endcsname\relax\else
\textbf{The length of the project description should not exceed 10 pages (font 11 pt, not including the references)!}
\vspace{0.5cm}

Note: Fill-out hints are given in italic. \textbf{Please remove them in the final version by removing the option \textbf{showhints} from the documentclass.}

\vspace{1cm}
Before using this template, please check at \url{https://go-nhr.de/computing-time} if a new version is available (this is version \templateversion).

\vspace{1cm}

When to use this template:
\begin{itemize}
  \item for \underline{\textbf{new NHR normal projects for already reviewed research projects}}: Depending on the NHR center you want to apply for (see list at \url{https://go-nhr.de/computing-time} you can use this template if you are applying for an NHR normal project and if your scientific project has already been successfully reviewed and approved by one of the accepted institutions (DFG, European Union, German Federal Ministries BM* (e.g. BMFTR; BMWK), Volkswagen Stiftung, another NHR or GCS center). 
\end{itemize}
When \textbf{not} to use this template:  
\begin{itemize}
  \item Don't use this template for \underline{\textbf{short-term extensions}} (up to three months). Contact your NHR center instead. 
  \item Don't use this template for NHR \underline{\textbf{large}} projects. 
  \item Don't use this template for \underline{\textbf{extensions}} of NHR normal and NHR large projects. For project extensions of projects, please use the simplified template at \url{https://go-nhr.de/computing-time}. 
  \item Don't use this template for normal projects if the center you want to apply for doesn’t allow the use of a simplified template for already reviewed research projects. The list can be found at \url{https://go-nhr.de/computing-time}. 
\end{itemize}
  \fi





\expandafter\ifx\csname whitelistedextensionproposal\endcsname\relax\else
\textbf{The length of the project description should not exceed 14 pages (font 11 pt, not including the references)!}
\vspace{0.5cm}

Note: Fill-out hints are given in italic. \textbf{Please remove them in the final version by removing the option \textbf{showhints} from the documentclass.}

\vspace{1cm}
Before using this template, please check at \url{https://go-nhr.de/computing-time} if a new version is available (this is version \templateversion).

\vspace{1cm}

When to use this template:
\begin{itemize}
  \item for \underline{\textbf{extension of an NHR normal project for already reviewed research projects}}: Depending on the NHR center you want to apply for (see list at \url{https://go-nhr.de/computing-time} you can use this template if you are applying for an NHR normal project and if your scientific project has already been successfully reviewed and approved by one of the accepted institutions (DFG, European Union, German Federal Ministries BM* (e.g. BMFTR; BMWK), Volkswagen Stiftung, another NHR or GCS center). 
\end{itemize}
When \textbf{not} to use this template:  
\begin{itemize}
  \item Don't use this template for NHR \underline{\textbf{large}} projects. 
  \item Don't use this template for \underline{\textbf{new}} NHR normal projects. 
  \item Don't use this template for normal projects if the center you want to apply for doesn’t allow the use of a simplified template for already reviewed research projects. The list can be found at \url{https://go-nhr.de/computing-time}. 
\end{itemize}
  \fi


%for all
In case of questions: If you are unsure or have questions please contact the center. The contacts can be found at \url{https://www.nhr-verein.de/unsere-mitglieder}. \\
The information provided here will be kept private. 
It will only be accessible to staff of the center you apply for, the resource allocation board of the center, the external scientific reviewers, the joint NHR resource allocation board (Nutzungsausschuss) which is a central NHR board that approves NHR large projects as well as the central NHR office (NHR Geschäftsstelle, for reporting purposes). The guidelines for confidentiality and neutrality of review processes apply. 
\endgroup
  \fi
  \clearpage
}


\newcommand{\Introduction}{
  \section{Introduction
  \expandafter\ifx\csname showhints\endcsname\relax\else
  \footnotesize{\color{nhr}(1 page text max., 0.5 recommended)}
  \fi
  }\label{sec:introduction}
  \expandafter\ifx\csname showhints\endcsname\relax\else
  \begingroup
  \itshape
  Give a short outline of the scientific background of your research, including references. The description should focus on the research rather than necessarily the HPC methodology which has been or is planned to be used.
  \endgroup
  
  \fi
}
\newcommand{\PreliminaryWork}{
  \section{Preliminary Work
  \expandafter\ifx\csname showhints\endcsname\relax\else
  \footnotesize{\color{nhr}(1 page text max., 0.5 recommended)}  
  \fi
  }\label{sec:preliminary_work}
  \expandafter\ifx\csname showhints\endcsname\relax\else
  \begingroup
  \itshape
  Provide a summary of your preliminary scientific and HPC-related work or projects connected to the proposed project, including references. 
  \endgroup

  \fi
}
\newcommand{\ProjectDescription}{
  \section{Description of the Project
  \expandafter\ifx\csname showhints\endcsname\relax\else
  \footnotesize{\color{nhr}(4-5 pages text max.)}
  \fi
  }\label{sec:project_description}
  \expandafter\ifx\csname showhints\endcsname\relax\else
  \begingroup
  \itshape
  Describe your research project in detail, structured in sub-projects, if applicable. Include a discussion of the scientific questions you plan to address and the overall scientific goals of the project. You must describe the proposal’s innovative aspects, impact and topicality. 
  \begin{itemize}
    \item Scientific objectives 
    \item Scientific and technical innovation potential and relevance 
  \end{itemize}
  \endgroup
  \fi
}

\newcommand{\ComputeResources}{
  \section{Compute Resources}\label{sec:compute_resources}
}

\newcommand{\SubprojectDescriptionHints}{
  \expandafter\ifx\csname SubprojectDescriptionHintsShown\endcsname\relax
  \begingroup
  \itshape
  Please discuss relevant aspects of the project. The following points serve as a guideline, but in cases where they do not apply to your project, they can be omitted as you see fit. 
  \begin{itemize}
    \item (Scientific objectives)
    \item (Scientific and technical innovation potential and relevance)
    \item Approach
    \item Computational objectives, types of calculations
    \item Which "types of runs" (see Chapter~\ref{sec:justification} Justification of Resources Requested) have you planned?
    \item Numerical methods and algorithms you want to use, improve, or design
    \item Progress beyond the state-of-the-art in terms of methods and calculations
    \item Expected outcome
  \end{itemize}
  \endgroup

  \newcommand\SubprojectDescriptionHintsShown{}
  \else
  ...
  \fi
}

\newcommand{\SubprojectDescription}[1]{
  \subsection{Sub-project \arabic{subsection}: #1}\label{sec:subproject\arabic{subsection}}
  \expandafter\ifx\csname showhints\endcsname\relax\else
  \SubprojectDescriptionHints
  \fi
}

\newcommand{\CodeOverview}{
  \subsection{Code Overview}\label{sec:code_overview}
  \expandafter\ifx\csname showhints\endcsname\relax\else
  \begingroup
  \itshape
  Please use the following table to provide information on up to \textbf{three} software packages you plan to use in the project that will consume a relevant portion (each >=10\%) of the requested compute time. 
  \endgroup
  \fi
}

\newenvironment{Codetable}[5]{
  \begin{center}
  \IfEqCase{#1}{
  {1}{ 
    \begin{tabular}{|L{#2}|L{#3}|}\hline
    \rowcolor{Gray}
    & Program A \\ \hline 
  }
  {2}{ 
    \begin{tabular}{|L{#2}|L{#3}|L{#4}|}\hline
    \rowcolor{Gray}
    & Program A & Program B \\ \hline 
  }
  {3}{ 
    \begin{tabular}{|L{#2}|L{#3}|L{#4}|L{#5}|}\hline
    \rowcolor{Gray}
    & Program A & Program B & Program C \\ \hline 
  }
  }[\PackageError{Codetable}{List at least one and at most three codes}{}]
}{
  \end{tabular}
  \end{center}
}

\newcommand{\CodeBenchmarks}{
  \subsection{Code Benchmarks}\label{sec:code_benchmarks}
  \expandafter\ifx\csname showhints\endcsname\relax\else
  \begingroup
  \itshape
The purpose of benchmarks in the project proposal is to demonstrate an efficient use of parallel resources and to estimate the resources needed. The benchmark case(s) should be representative for your workload(s). 
      \vspace{0.5cm}

Even workloads with unfavorable parallel scaling can be granted computation time if the HPC systems offered by NHR are required to achieve progress for the given scientific problem that would not be possible with other HPC systems. The same holds for embarrassingly parallel workloads (high-throughput computing, e.g., parameter studies ) where the total computational need requires a large HPC system offered by NHR. In these cases, no scalability benchmarks are needed. 
      \vspace{0.5cm}

      Fill out the benchmark section for each program listed section~\ref{sec:code_overview}. Please refer to example evaluations on our website \url{https://go-nhr.de/computing-time}. The NHR centers offer NHR-test/NHR-preparation and NHR-starter projects that you can use to benchmark/test your programs on the desired NHR-center. 

      Some NHR centers offer a program-whitelisting procedure, meaning that no benchmarks/scaling tests are required for those codes and the following sections can be skipped. The lists can be found at \url{https://go-nhr.de/computing-time}. 

If you want to provide an alternative demonstration (e.g. performance model) of parallel performance, please contact the respective center beforehand. 

 
  \endgroup
  \fi
}

\newcommand{\Benchmark}[1]{
  \subsubsection{Program \Alph{subsubsection}: #1
  \expandafter\ifx\csname showhints\endcsname\relax\else
  \footnotesize{\color{nhr}(1 page text max. per relevant code, 0.5 recommended)}  
  \fi
  }\label{sec:benchmark\Alph{subsubsection}}
}

\newcommand{\BenchmarkDescriptionHints}{
  \expandafter\ifx\csname BenchmarkDescriptionHintsShown\endcsname\relax
  \begingroup
  \itshape
  Please provide a brief description of the calculations performed. If program A is used for a wide range of problem sizes (e.g., a factor of 10 or greater between problem sizes), then the benchmarks should cover the full range of problem sizes (PS1, ...). 
      \begin{itemize}
    \item How does the benchmark problem size relate to the problem size in the respective subprojects? 
    \item List the relevant execution parameters of the benchmark jobs (e.g., MPI ranks per node, OpenMP threads per MPI rank, ...) 
    \item How is the code performance calculated? (runtime, TFLOPS,...) 
  \end{itemize}
  \endgroup
  \newcommand\BenchmarkDescriptionHintsShown{}
  \fi
}

\newcommand\BenchmarkDescription{
  \paragraph{Benchmark Description}
  \expandafter\ifx\csname showhints\endcsname\relax\else
    \BenchmarkDescriptionHints
  \fi
}
  
\newcommand{\BenchmarkSystemHints}{
  \expandafter\ifx\csname BenchmarkSystemHintsShown\endcsname\relax
  \begingroup
  \itshape
  Please describe the computing systems used for the benchmarks. Typically, only benchmarks on one computing system are needed for the planned calculation. This compute system should be sufficiently similar to the resources/computing system requested in this proposal. Describe them all, if different systems are compared (e.g., to justify specific hardware requests like GPU hours instead of CPU hours). 
  \endgroup
  \newcommand\BenchmarkSystemHintsShown{}
  \fi
}

\newcommand\BenchmarkSystem{
  \paragraph{Benchmark System}
  \expandafter\ifx\csname showhints\endcsname\relax\else
    \BenchmarkSystemHints
  \fi
}

\newcommand{\BenchmarkDataHints}{
  \expandafter\ifx\csname BenchmarkDataHintsShown\endcsname\relax
  \begingroup
  \itshape
Please present scaling plots as performance (e.g., steps/time, ns/day) on the y-axis against over used compute resources (e.g., cores, nodes) on the x-axis. Include a reference line for theoretical ideal scaling for a quick assessment of the parallel efficiency. Please include separate graphs for each relevant problem size (PS1,...). 

  Alternatively, show a scaling table (performance vs. used compute resources and ideal scaling). 
  \endgroup
  \newcommand\BenchmarkDataHintsShown{}
  \fi
}

\newcommand\BenchmarkData{
  \paragraph{Scaling Plot or Scaling Table}
  \expandafter\ifx\csname showhints\endcsname\relax\else
    \BenchmarkDataHints
  \fi
}
  
\newcommand{\BenchmarkDiscussionHints}{
  \expandafter\ifx\csname BenchmarkDiscussionHintsShown\endcsname\relax
  \begingroup
  \itshape
  Discuss how the benchmarks inform your choice of parallel resources for the subprojects, and how that choice impacts your resource calculation. Do this for all problem sizes and different parallel resources.  
  \endgroup
  \newcommand\BenchmarkDiscussionHintsShown{}
  \fi
}

\newcommand\BenchmarkDiscussion{
  \paragraph{Discussion, Relevance to the Planned Calculations}
  \expandafter\ifx\csname showhints\endcsname\relax\else
    \BenchmarkDiscussionHints
  \fi
}

\newcommand\ResourceJustification{
  \subsection{Justification of Resources Requested}\label{sec:justification}
  \expandafter\ifx\csname showhints\endcsname\relax\else
    \begingroup
    \itshape
Outline the amount of resources you request for the current proposal, structured in sub-projects, if applicable. For multiyear projects (not extensions), all years must be included. Please explain the \# runs (R1,...) in the context of the subproject goals. 

If you want to request resources not listed in the table, like FPGAs or other devices, please add them analogous to GPUs in the table below. 

    \begin{description}
      \item [Run type:] A one or two-word description of the calculation relating to the subprojects described above
      \item [Program used:] The software used for the calculation should be listed in section~\ref{sec:code_overview}.
      \item [Problem size (PS1,\ldots):] A metric of the investigated system that relates to the computing cost, e.g., number of atoms, number of cells, number of electrons, degrees of freedom 
      \item [\# runs (R1,\ldots):] The number of distinct calculation objectives you plan for this type and subproject, e.g., number of different starting configurations, different temperatures, different molecules
      \item [\# steps per run (S1,\ldots):] Number of calculation steps needed for the planned run. This can be an estimate if the calculation is ended by e.g., convergence criteria or a fixed number if a step limit ends the calculation. If your simulation is not iterative, specify one here and the job runtime as wall time per step. 
      \item [Wall time per step (W1,\ldots):] Calculation time required per step (S1,\ldots) with the planned CPU-cores (C1,\ldots) or GPUs (G1,\ldots) per run. If a different metric better fits your situation, change as necessary. 
      \item [\# CPU-cores / run (C1,\ldots):] Number of CPU-cores you plan to use in this kind of calculation. This must be informed by the benchmarking of the respective program. 
      \item [\# GPUs / run (G1,\ldots):] Number of GPUs you plan to use in this calculation. This must be informed by the benchmarking of the respective program. Please note that you will also need some CPU-resources for GPU-jobs. 
    \end{description}
    For molecular dynamics and similar simulations you can use the table ResourceTableMD instead, that uses
    \begin{description}
      \item [Simulation time in $\mu$s (T1,\ldots):] The desired simulation time in $\mu$s
      \item [Performance in ns/day (P1,\ldots):] The performance of the molecular dynamics simulation in ns/day with the chosen resources and workload.
    \end{description}
    and automatically calculates.

    In case none of the automatic tables fit your situation, you can find a template for a manual table in the tex file.
    \endgroup
  \newcommand\ResourceJustificationHintsShown{}
  \fi
}

\newcounter{RTres}
\newcounter{RTncol}
\newcommand\RTtype{0}
\newcommand\RTtabsep{0.1}
\newcommand\summe{0}
\newcounter{subproj}
\newcounter{itype}
\newcounter{res}
\newcounter{mon}
\newcounter{PTncol}
\newcommand\PTtabwidth{15}
\newcommand\PTmonthwidth{2}
\newcounter{PTntypes}
\newcommand\PTcolwidth{0}
\newcommand\PTtabsep{0.1}
\newcommand\resfactor{0}

\newcommand{\ResourceTable}[2]{
  \subsubsection{#1 Resources:}
  \setsepchar{\\/;}
  \readlist\args{#2}
  \renewcommand\RTtype{#1}
  \setcounter{RTncol}{9}

  %set widths

  \setlength{\tabcolsep}{\RTtabsep cm}
  \setlength{\doublerulesep}{0.0mm}
  \IfSubStr{#1}{CPU}{\FPeval{\resfactor}{0.000001}}{\FPeval{\resfactor}{0.001}}
  
  %temp vars
  \FPeval{\summe}{0}
   \forloop{subproj}{1}{\value{subproj}<\argslen}{
     \FPeval{\summe}{\summe+\fpeval{\args[\value{subproj},8]*\args[\value{subproj},7]*\args[\value{subproj},6]*\args[\value{subproj},5]*\resfactor}}
   }

  \begin{tabular}{|p{1.3cm}|p{2cm}|p{2cm}|p{2cm}|p{1.2cm}|p{1.2cm}|p{1.3cm}|p{1.2cm}|p{1.7cm}|} \hline
    \rowcolor{Gray}    Sub-project & Run type & Programs used &  Problem size & \#runs & \#steps per run & Wall time per step [hours] & \IfSubStr{#1}{CPU}{\#CPU-cores / run}{\##1s / run} & Total \reslabel{#1}\\ \hline
  \expandafter\ifx\csname showhints\endcsname\relax\else
  \IfSubStr{#1}{CPU}{
  1 & Preproc & program A & $PS1$ & $R1$ & $S1$ & $W1$ & $C1$ & $R1\cdot S1\cdot W1\cdot C1\cdot 10^{-6}$ \\ \hline
  ... & Type 1 & program B & $PS2$ & $R2$ & $S2$ & $W2$ & $C2$ & $R2\cdot S2\cdot W2\cdot C2\cdot 10^{-6}$ \\ \hline
    }{
1 & Preproc & program A & $PS1$ & $R1$ & $S1$ & $W1$ & $G1$ & $R1\cdot S1\cdot W1\cdot G1\cdot 10^{-3}$ \\ \hline
... & Type 1 & program B & $PS2$ & $R2$ & $S2$ & $W2$ & $G2$ & $R2\cdot S2\cdot W2\cdot G2\cdot 10^{-3}$ \\ \hline
    }
    \fi
   \forloop{subproj}{1}{\value{subproj}<\argslen}{
     \args[\value{subproj},1] & \args[\value{subproj},2] & \args[\value{subproj},3] & \args[\value{subproj},4] & \args[\value{subproj},5] & \args[\value{subproj},6] & \args[\value{subproj},7] & \args[\value{subproj},8] & 
     \fpeval{\args[\value{subproj},8]*\args[\value{subproj},7]*\args[\value{subproj},6]*\args[\value{subproj},5]*\resfactor} 
       \ifthenelse{\equal{\fpeval{\value{subproj}}}{\fpeval{\argslen-1}}}{\\ \hlinewd{1.5pt}}{\\ \hline}
     } 
    TOTAL& - & - & -& -&-&-&-& \fpeval{\summe} \\ \hline 
  \end{tabular}
}

\newcommand{\ResourceTableMD}[2]{
  \subsubsection{#1 Resources:}
  \setsepchar{\\/;}
  \readlist\args{#2}
  \renewcommand\RTtype{#1}
  \setcounter{RTncol}{9}

  %set widths

  \setlength{\tabcolsep}{\RTtabsep cm}
  \setlength{\doublerulesep}{0.0mm}
  \IfSubStr{#1}{CPU}{\FPeval{\resfactor}{0.024}}{\FPeval{\resfactor}{24}}
  
  %temp vars
  \FPeval{\summe}{0}
   \forloop{subproj}{1}{\value{subproj}<\argslen}{
     \FPeval{\summe}{\summe+\fpeval{\args[\value{subproj},8]/\args[\value{subproj},7]*\args[\value{subproj},6]*\args[\value{subproj},5]*\resfactor}}
   }

  \begin{tabular}{|p{1.3cm}|p{2cm}|p{2cm}|p{2cm}|p{1.2cm}|p{1.2cm}|p{1.3cm}|p{1.2cm}|p{1.7cm}|} \hline
    \rowcolor{Gray}    Sub-project & Run type & Programs used &  Problem size & \#runs & Sim. time in $\mu$s & Perf. in ns/day & \IfSubStr{#1}{CPU}{\#CPU-cores / run}{\##1s / run} & Total \reslabel{#1}\\ \hline
  \expandafter\ifx\csname showhints\endcsname\relax\else
  \IfSubStr{#1}{CPU}{
    1 & Preproc & program A & $PS1$ & $R1$ & $T1$ & $P1$ & $C1$ & $R1\cdot T1/P1\cdot C1\cdot 24\cdot 10^{-3}$ \\ \hline
  ... & Type 1 & program B & $PS2$ & $R2$ & $T2$ & $P2$ & $C2$ & $R2\cdot T2/P2\cdot C2\cdot 24\cdot 10^{-3}$ \\ \hline
    }{
1 & Preproc & program A & $PS1$ & $R1$ & $T1$ & $P1$ & $G1$ & $R1\cdot T1/P1\cdot G1\cdot 24$ \\ \hline
... & Type 1 & program B & $PS2$ & $R2$ & $T2$ & $P2$ & $G2$ & $R2\cdot T2/P2\cdot G2\cdot 24$ \\ \hline
    }
    \fi
   \forloop{subproj}{1}{\value{subproj}<\argslen}{
     \args[\value{subproj},1] & \args[\value{subproj},2] & \args[\value{subproj},3] & \args[\value{subproj},4] & \args[\value{subproj},5] & \args[\value{subproj},6] & \args[\value{subproj},7] & \args[\value{subproj},8] & 
     \fpeval{round(\args[\value{subproj},8]/\args[\value{subproj},7]*\args[\value{subproj},6]*\args[\value{subproj},5]*\resfactor,3)} 
       \ifthenelse{\equal{\fpeval{\value{subproj}}}{\fpeval{\argslen-1}}}{\\ \hlinewd{1.5pt}}{\\ \hline}
     } 
    TOTAL& - & - & -& -&-&-&-& \fpeval{round(\summe,3)} \\ \hline 
  \end{tabular}
}

\newcommand\ResourcePlan{
  \section{Resource Management and Work Schedule
  \expandafter\ifx\csname showhints\endcsname\relax\else
  \footnotesize{\color{nhr}(0.5 pages per year)}
  \fi
  }\label{sec:resourceplan}
  \expandafter\ifx\csname showhints\endcsname\relax\else
    \begingroup
    \itshape
    Describe the planned resource usage by filling out the table below. In case of a multiyear project proposal (not an extension), please present one table per year. If you plan a roughly constant usage over time, you can skip the table(s) and state that a continuous usage is intended. 
    \endgroup
  \fi
}

\newcommand{\PlanTable}[3]{
  \setsepchar{\\/;/,}
  \readlist\args{#3}
  
  \renewcommand\PTtabwidth{#1}

  %count resource columns
  \setcounter{PTncol}{0}
  \forloop{subproj}{1}{\value{subproj}<\argslen}{
    \forloop{res}{2}{\value{res}<\listlen\args[\value{subproj}]}{
      \addtocounter{PTncol}{1}
    }
  }
  %count resource types
  \setsepchar{,}
  \readlist\PTtypes{#2}
  \setcounter{PTntypes}{0}

  \forloop{itype}{0}{\value{itype}<\PTtypeslen}{
      \addtocounter{PTntypes}{1}
  }

  %set widths
  \FPeval{\PTcolwidth}{(\PTtabwidth-\PTmonthwidth)/(\arabic{PTncol}+\arabic{PTntypes})-\PTtabsep}
  \FPeval{\PTcolwidthb}{\PTcolwidth-0*\PTtabsep}

  \setlength{\tabcolsep}{\PTtabsep cm}
  \setlength{\doublerulesep}{0.0mm}
  

  \begin{tabular}{|p{\PTmonthwidth cm}|*{\arabic{PTncol}}{p{\PTcolwidthb cm}|}|*{\arabic{PTntypes}}{p{\PTcolwidth cm}|} } \hline
    \rowcolor{Gray}Project Months \forloop{subproj}{1}{\value{subproj}<\argslen}{ & \multicolumn{\fpeval{\listlen\args[\value{subproj}]-2}}{p{\fpeval{(\PTcolwidth+\PTtabsep*2)*(\listlen\args[\value{subproj}]-2)} cm}|}{\cellcolor{Gray}\args[\value{subproj},1,1]}\hspace{-0.257cm} }
         & \multicolumn{\arabic{PTntypes}}{|l|}{Sum}
       \\ \hline
       \forloop{subproj}{1}{\value{subproj}<\argslen}{
         \forloop{res}{2}{\value{res}<\listlen\args[\value{subproj}]}
         {
            & \args[\value{subproj},\value{res},1] \reslabel{\args[\value{subproj},\value{res},1]}
         }
       }
       \forloop{itype}{1}{\value{itype}<\fpeval{1+\PTtypeslen}}{
         & \PTtypes[\value{itype}] \reslabel{\PTtypes[\value{itype}]}
       }\\ \hline
     \forloop{mon}{1}{\value{mon}<13}{
       \arabic{mon} 
       \forloop{subproj}{1}{\value{subproj}<\argslen}
       {
         \forloop{res}{2}{\value{res}<\listlen\args[\value{subproj}]}
         {
           &\args[\value{subproj},\value{res},\fpeval{\value{mon}+1}]
         } 
       }
       %row sums
        \forloop{itype}{1}{\value{itype}<\fpeval{1+\PTtypeslen}}{
          & \FPeval{\summe}{0}
          \forloop{subproj}{1}{\value{subproj}<\argslen}
          {
            \forloop{res}{2}{\value{res}<\listlen\args[\value{subproj}]}
            {
              \IfSubStr{\args[\value{subproj},\value{res},1]}{\PTtypes[\value{itype}]}{
              \FPeval{\summe}{\summe+\fpeval{\args[\value{subproj},\value{res},\fpeval{\value{mon}+1}]}}
            }{}
            }
          }
          \fpeval{\summe}
        }
       \ifthenelse{\equal{\fpeval{\value{mon}}}{3}}{\\ \hlinewd{1.5pt}}{
        \ifthenelse{\equal{\fpeval{\value{mon}}}{6}}{\\ \hlinewd{1.5pt}}{
          \ifthenelse{\equal{\fpeval{\value{mon}}}{9}}{\\ \hlinewd{1.5pt}}{
            \ifthenelse{\equal{\fpeval{\value{mon}}}{12}}{\\ \hlinewd{1.5pt}}{\\ \hline}
          }
        }
      }

    } 
    %column sums
    Sum 
       \forloop{subproj}{1}{\value{subproj}<\argslen}
       {
         \forloop{res}{2}{\value{res}<\listlen\args[\value{subproj}]}
         {&
          \FPeval{\summe}{0}
          \forloop{mon}{2}{\value{mon}<14}{
            \FPeval{\summe}{\summe+\fpeval{\args[\value{subproj},\value{res},\value{mon}]}}
          } 
          \fpeval{\summe}
        }
    }
    % total sums
        \forloop{itype}{1}{\value{itype}<\fpeval{1+\PTtypeslen}}{
          & \FPeval{\summe}{0}
          \forloop{mon}{1}{\value{mon}<13}
          {
            \forloop{subproj}{1}{\value{subproj}<\argslen}
            {
              \forloop{res}{2}{\value{res}<\listlen\args[\value{subproj}]}
              {
                \IfSubStr{\args[\value{subproj},\value{res},1]}{\PTtypes[\value{itype}]}{
                \FPeval{\summe}{\summe+\fpeval{\args[\value{subproj},\value{res},\fpeval{\value{mon}+1}]}}
              }{}
              }
            }
          }
          \fpeval{\summe}
        }
    \\ \hline 
  \end{tabular}
}

\newcommand{\SpecialRequirements}{
  \section{Special Requirements (optional)
  \expandafter\ifx\csname showhints\endcsname\relax\else
  \footnotesize{\color{nhr}(maximum 0.5 page text)}
  \fi
  }\label{sec:special_requirements}
  \expandafter\ifx\csname showhints\endcsname\relax\else
  \begingroup
  \itshape
  \begin{itemize}
    \item Will your application benefit from FPGAs or GPUs?
    \item What is the major bottleneck in your current use of HPC systems?
    \item Is there any special need for large-scale pre- or postprocessing? 
    \item Do you have any special requirements for your workflow?
    \item Do you require exceptional capacities for data transfer or data storage? 
    \item Do you need help with a data management plan? 
    \item Could we support you with our consulting offers or our scientific support? 
  \end{itemize}
  \endgroup
  \fi
}


\newcommand{\References}[1]{
  \section{References}\label{sec:references}
  \expandafter\ifx\csname showhints\endcsname\relax\else
  \begingroup
  \itshape
  You can place your reference in the bibtext file #1.bib and cite them in this proposal as \cite{PhysRev.140.A1133}. For more details on references in bibtex, have a look at \url{https://www.overleaf.com/learn/latex/Bibliography_management_with_bibtex}. Also, feel free to use other mechanisms like biblatex (see \url{https://de.overleaf.com/learn/latex/Bibliography_management_with_biblatex}) if you prefer them.
  \endgroup
  \fi
  \begingroup
%  \refname{}
  \renewcommand{\bibsection}{}%
  \bibliographystyle{plainnat}
  \bibliography{#1} 
  \endgroup
}



\newcommand{\ProgressReport}{
  \section{Progress Report}\label{sec:progress_report}
}

\newcommand{\ProgressReportSummary}{
  \subsection{Short Summary
  \expandafter\ifx\csname showhints\endcsname\relax\else
  \footnotesize{\color{nhr}(about 0.5 page)}
  \fi
  }\label{sec:progress_report}
  \expandafter\ifx\csname showhints\endcsname\relax\else
  \begingroup
  \itshape
Shortly summarize the progress of the project in relation to the previous project proposal. 

Please list here explicitly your publications that were using the HPC resources in the previous project period. 
  \endgroup
  \fi
}

\newcommand{\SubprojectReport}[1]{
  \subsection{Sub-project \arabic{subsection}: #1
  \expandafter\ifx\csname showhints\endcsname\relax\else
  \footnotesize{\color{nhr}(max. 1 page text per sub-project)}
  \fi
  }\label{sec:subproject\arabic{subsection}}
  \expandafter\ifx\csname showhints\endcsname\relax\else
  \begingroup
  \itshape
  \endgroup

  \fi
}

\newcommand{\SubprojectReportProgress}{
  \subsubsection{Project Progress}
  \expandafter\ifx\csname showhints\endcsname\relax\else
  \begingroup
  \itshape
  Shortly describe the progress and accomplishments in the sub-project here including references to your publications.  
  \endgroup
  \fi
}

\newcommand{\SubprojectReportChallenges}{
  \subsubsection{Project Challenges}
  \expandafter\ifx\csname showhints\endcsname\relax\else
  \begingroup
  \itshape
  Please describe how the scientific challenges of the sub-project were tackled and solved here. 
  \endgroup
  \fi
}

\newcommand{\SubprojectReportProblems}{
  \subsubsection{Problems Encountered (optional)}
  \expandafter\ifx\csname showhints\endcsname\relax\else
  \begingroup
  \itshape
  Please describe the technical, scientific or organizational problems encountered during the sub-project duration here. How did they influence the usage of your granted resource contingent? 
  \endgroup
  \fi
}

\newcommand{\ProgressImprovements}{
  \section{Suggestions for Improvements of Services (optional)}
  \expandafter\ifx\csname showhints\endcsname\relax\else
  \begingroup
  \itshape
  Please let us know here which hardware, software or services would have been helpful for your project or how else we could improve for you as a researcher.  
  \endgroup
  \fi
}

\newcommand{\ProjectExtensionDescription}{
  \section{Description of the Project Extension}\label{sec:project_description}
  \expandafter\ifx\csname showhints\endcsname\relax\else
  \begingroup
  \itshape
  Important: Please describe here all sub projects for which you want to request compute time in the extension. Descriptions of sub projects that are identical to the previous project phase can be copied. 
  \endgroup
  \fi
}

\newcommand{\SubprojectExtensionDescriptionHints}{
  \expandafter\ifx\csname SubprojectExtensionDescriptionHintsShown\endcsname\relax
  \begingroup
  \itshape
  Please discuss relevant aspects of the project. The following points serve as a guideline, but in cases where they do not apply to your project, they can be omitted as you see fit. 
  \begin{itemize}
    \item Scientific objectives
    \item Scientific and technical innovation potential and relevance
    \item Approach
    \item Computational objectives, types of calculations
    \item Which "types of runs" (see Chapter~\ref{sec:justification} Justification of Resources Requested) have you planned?
    \item Numerical methods and algorithms you want to use, improve, or design
    \item Progress beyond the state-of-the-art in terms of methods and calculations
    \item Expected outcome
  \end{itemize}
  \endgroup

  \newcommand\SubprojectExtensionDescriptionHintsShown{}
  \else
  ...
  \fi
}

\newcommand{\SubprojectExtensionDescription}[1]{
  \subsection{Sub-project \arabic{subsection}: #1
  \expandafter\ifx\csname showhints\endcsname\relax\else
  \footnotesize{\color{nhr}(1 page text max. per sub-project, 0.5 recommended)}
  \fi
  }\label{sec:subproject\arabic{subsection}}
  \expandafter\ifx\csname showhints\endcsname\relax\else
  \SubprojectExtensionDescriptionHints
  \fi
}



\newcommand{\ComputeResourcesExtension}{
  \section{Compute Resources}\label{sec:compute_resources}
  \textbf{Important note for applicants and reviewers: This section is only required in case there are relevant changes in the programs to be used or the properties of calculations compared to the previous proposal (e.g. relevant changes in system sizes that influences the scaling behavior). Otherwise remove sections~\ref{sec:code_overview} and~\ref{sec:code_benchmarks}.}
  \expandafter\ifx\csname showhints\endcsname\relax\else
  \begingroup
  \itshape
  \endgroup
  \fi
}

\newcommand{\ShortProjectDescription}{
  \section{Short Description of the Project}\label{sec:project_description}
  \expandafter\ifx\csname showhints\endcsname\relax\else
  \begingroup
  \itshape
  Shortly list the simulations planned in the project and how they relate to the research topic of the already reviewed research project. The description should contain the information needed so that a researcher from your field can judge if the requested HPC resources are suitable. A division into subprojects is optional. 
  \endgroup
  \fi
}

\newcommand{\ShortSubprojectDescriptionHints}{
  \expandafter\ifx\csname ShortSubprojectDescriptionHintsShown\endcsname\relax
  \begingroup
  \itshape
  \endgroup

  \newcommand\ShortSubprojectDescriptionHintsShown{}
  \else
  ...
  \fi
}

\newcommand{\ShortProjectExtensionDescription}{
  \section{Short Description of the Project Extension
  \expandafter\ifx\csname showhints\endcsname\relax\else
  \footnotesize{\color{nhr}(0.5 page text max. per sub-project)}
  \fi
  }\label{sec:project_description}
  \expandafter\ifx\csname showhints\endcsname\relax\else
  \begingroup
  \itshape
  Shortly list the simulations planned in the project and how they relate to the research topic of the already reviewed research project. The description should contain the information needed so that a researcher from your field can judge if the requested HPC resources are suitable. A division into subprojects is optional. 
  \endgroup
  \fi
}

\newcommand{\ShortSubprojectExtensionDescriptionHints}{
  \expandafter\ifx\csname ShortSubprojectExtensionDescriptionHintsShown\endcsname\relax
  {\color{nhr}(0.5 page text max. per sub-project)} 
  \newcommand\ShortSubprojectDescriptionHintsShown{}
  \else
  ...
  \fi
}

\newcommand{\ShortSubprojectDescription}[1]{
  \subsection{Sub-project \arabic{subsection}: #1
  \expandafter\ifx\csname showhints\endcsname\relax\else
  \footnotesize{\color{nhr}(0.5 page text max. per sub-project)}
  \fi
  }\label{sec:subproject\arabic{subsection}}
  \expandafter\ifx\csname showhints\endcsname\relax\else
  \ShortSubprojectDescriptionHints
  \fi
}
