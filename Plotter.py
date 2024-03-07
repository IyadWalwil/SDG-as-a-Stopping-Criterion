import matplotlib as mpl
from matplotlib import pyplot as plt
from cycler import cycler
mpl.rcParams['axes.prop_cycle'] = cycler(color='brgmcyk')

class Plotter:
    """
    A class for plotting curves using matplotlib. It basically takes a dictionary of curves along with their labels and titles. 

    Methods:
        - measures_dict: Constructs a dictionary of optimization measures along with their titles and labels.
        - OG_dict: Constructs a dictionary for visualizing Optimality Gap and its bounds.
        - KKT_SDG_dict: Constructs a dictionary for visualizing Karush–Kuhn–Tucker error and Smoothed Duality Gap bounds.
        - PDG_SDG_dict: Constructs a dictionary for visualizing Projected Duality Gap and Smoothed Duality Gap bounds.
        - plot: Plots the given data dictionary.
        - findings_plot: Plots our findings.
    """
    def measures_dict(self, OG, FG, KKT, PDG, SDG):
        """
        Constructs a dictionary of optimization measures along with their titles and labels.

        Parameters:
            OG: Optimality gap.
            FG: Feasibility gap.
            KKT: Karush–Kuhn–Tucker error.
            PDG: Projected duality gap.
            SDG: Smoothed duality gap.

        Returns:
            dict: A dictionary of dictionaries containing optimization measures along with their titles and labels.
                  Such that: 
                    - The key of the dictionary represents the title of the measure. 
                    - The value of the dictionary represents a sub-dictionary.
                    - The key of a sub-dictionary represents the label of a measure. 
                    - The value of a sub-dictionary represents a measure itself.
        """
        dict = {'Optimality gap': {"$\mathcal{V}(x)$": OG},
                'Feasibility gap': {"$\mathcal{F}(x)$": FG},
                'Optimality gap and Feasibility error': {"$\mathcal{V}(x) + \mathcal{F}(x)$": OG+ FG},
                'Karush–Kuhn–Tucker error': {"$\mathcal{K}(z)$": KKT}, 
                'Projected duality gap': {"$\mathcal{D}(z)$": PDG}, 
                'Smoothed duality gap': {"$\mathcal{G}(z)$": SDG}
        }
        return dict
    
    def OG_dict(self, OG, FG, OG_bounds):
        """
        Constructs a dictionary for visualizing Optimality Gap and its bounds.

        Parameters:
            OG: Optimality gap.
            FG: Feasibility gap.
            OG_bounds: Optimality gap bounds.

        Returns:
            dict: A dictionary of dictionaries containing the Optimality Gap bounds along with their titles and labels.
                  Such that: 
                    - The key of the dictionary represents the title of the plot. 
                    - The value of the dictionary represents a sub-dictionary.
                    - The key of a sub-dictionary represents the label of a curve (bound, measure). 
                    - The value of a sub-dictionary represents the bound (measure) itself.
        """
        OGFG = OG + FG
        dict = {"Theorem 2: KKT approximation for OG": {"Optimality gap": OGFG,
                                                         'KKT bound': OG_bounds['KKT']+FG
                                                         },
                "Theorem 3: SDG approximation for OG": {"Optimality gap": OGFG, 
                                                         'SDG bound': OG_bounds['SDG']+FG
                                                         },
                "Theorem 4: PDG approximation for OG": {"Optimality gap": OGFG, 
                                                         'PDG bound': OG_bounds['PDG']+FG
                                                         },
                "Optimality gap vs. its bounds": {"Optimality gap": OGFG,
                                                  "KKT bound": OG_bounds['KKT']+FG,
                                                  "SDG bound": OG_bounds['SDG']+FG,
                                                  "PDG bound": OG_bounds['PDG']+FG
                                                  }
                }
        return dict    
    
    def KKT_SDG_dict(self, KKT, KKT_SDG_bounds):
        """
        Constructs a dictionary for visualizing Karush–Kuhn–Tucker error and Smoothed Duality Gap bounds.

        Parameters:
            KKT: Karush–Kuhn–Tucker error.
            KKT_SDG_bounds: KKT and SDG approximations of each other.

        Returns:
            dict: A dictionary of dictionaries containing the KKT and SDG bounds along with their titles and labels.
                  Such that: 
                    - The key of the dictionary represents the title of the plot. 
                    - The value of the dictionary represents a sub-dictionary.
                    - The key of a sub-dictionary represents the label of a curve (bound, measure). 
                    - The value of a sub-dictionary represents the bound (measure) itself.
        """
        dict = {"Theorem 5: KKT approximation for SDG": {"$\mathcal{G}(z)$": KKT_SDG_bounds["G<K"]['SDG'], 
                                                         '$β̲ \mathcal{K}(z)$': KKT_SDG_bounds["G<K"]['bound']
                                                         },
                "Theorem 6: SDG approximation for KKT": {"$\mathcal{K}(z)$": KKT, 
                                                          '$\\bar{\\beta}_L \mathcal{G}_{\\beta}(z)$': KKT_SDG_bounds["K<G"]
                                                          }
        }
        return dict

    def PDG_SDG_dict(self, PDG, PDG_SDG_bounds):
        """
        Constructs a dictionary for visualizing Projected Duality Gap and Smoothed Duality Gap bounds.

        Parameters:
            PDG: Projected duality gap.
            PDG_SDG_bounds: PDG and SDG approximations of each other.

        Returns:
            dict: A dictionary of dictionaries containing the PDG and SDG bounds along with their titles and labels.
                  Such that: 
                    - The key of the dictionary represents the title of the plot. 
                    - The value of the dictionary represents a sub-dictionary.
                    - The key of a sub-dictionary represents the label of a curve (bound, measure). 
                    - The value of a sub-dictionary represents the bound (measure) itself.
        """
        dict = {"Theorem 7: PDG approximation for SDG": {"$\mathcal{G}(z)$": PDG_SDG_bounds["G<D"]['SDG'], 
                                                         'PDG bound': PDG_SDG_bounds["G<D"]['bound']
                                                         },
                "Theorem 8: SDG approximation for PDG": {"$\mathcal{D}(z)$": PDG, 
                                                          'SDG bound': PDG_SDG_bounds["D<G"]
                                                          }
        }
        return dict
    
    def paper_plots_dicts(self, OG, FG, KKT, PDG, OG_bounds, KKT_SDG_bounds, PDG_SDG_bounds):
        one_dim_dict = {"Fig. 3 - (a)": {"$\mathcal{K}(z)$": KKT, 
                                        "$\\bar{\\beta}_L \mathcal{G}_{\\beta}(z)$": KKT_SDG_bounds["K<G"]},
                        "Fig. 7 - (a)": {"SDG": PDG_SDG_bounds["G<D"]['SDG'], 
                                         'PDG bound': PDG_SDG_bounds["G<D"]['bound']},
                        "Fig. 8 - (a)": {"PDG": PDG, 
                                         'SDG bound': PDG_SDG_bounds["D<G"]}
                        }
        IID_dict = {"Fig. 3 - (b)": {"$\mathcal{K}(z)$": KKT, 
                                    "$\\bar{\\beta}_L \mathcal{G}_{\\beta}(z)$": KKT_SDG_bounds["K<G"]}} 
        cov_dict = {"Fig. 3 - (c)": {"$\mathcal{G}(z)$": KKT_SDG_bounds["G<K"]['SDG'], 
                                     '$β̲ \mathcal{K}(z)$': KKT_SDG_bounds["G<K"]['bound']},
                    "Fig. 7 - (b)": {"SDG": PDG_SDG_bounds["G<D"]['SDG'], 
                                    'PDG bound': PDG_SDG_bounds["G<D"]['bound']},
                    "Fig. 8 - (b)": {"PDG": PDG, 
                                     'SDG bound': PDG_SDG_bounds["D<G"]}
                    }
        distributed_dict = {"Fig. 4": {"Optimality gap": OG + FG,
                                                  "KKT bound": OG_bounds['KKT']+FG,
                                                  "SDG bound": OG_bounds['SDG']+FG,
                                                  "PDG bound": OG_bounds['PDG']+FG
                                                  }}
        QP_dict = {"Fig. 5 - (a)": {"SDG": PDG_SDG_bounds["G<D"]['SDG'], 
                                    "PDG bound": PDG_SDG_bounds["G<D"]['bound']},
                   "Fig. 5 - (b)": {"PDG": PDG, 
                                    'SDG bound': PDG_SDG_bounds["D<G"]}
                    }
        BP_dict = {"Fig. 7 - (c)": {"SDG": PDG_SDG_bounds["G<D"]['SDG'], 
                                    "PDG bound": PDG_SDG_bounds["G<D"]['bound']},
                   "Fig. 8 - (c)": {"PDG": PDG, 
                                    'SDG bound': PDG_SDG_bounds["D<G"]}
                    }
        return one_dim_dict, IID_dict, cov_dict, distributed_dict, QP_dict, BP_dict

    def plot(self, data_dict, ylabel='Value of the plotted measures', 
                    xlabel='Iteration', markers_list = [None, '^', 's', 'o', 'x', '*'], 
                    show_titles=True, labels_out = False, labels_loc = 'upper right'):
        """
        Plots the given data dictionary.

        Parameters:
            data_dict (dict): A dictionary containing data to plot.
            ylabel (str): y-axis label.
            xlabel (str): x-axis label.
            markers_list (list): List of markers for different plots.
            show_titles (bool): Whether to show titles for plots.
            labels_out (bool): Whether to put legends outside the plot.
            labels_loc (str): Location of legends.
        """
        plt.figure()
        for title, sub_dict in data_dict.items():
            for (label, fun), marker in zip(sub_dict.items(), markers_list):
                plt.semilogy(fun, label=label, marker = marker, markevery = max(1, (len(fun)//20)))
            if show_titles:
                plt.title(title, fontsize=15)
            plt.xlabel(xlabel, fontsize=14)
            plt.ylabel(ylabel, fontsize=13)
            plt.grid(which='both')
            if labels_out:
                plt.legend(loc='center left', fancybox=True, prop={'size': 12}, framealpha=0.4, bbox_to_anchor=(1, 0.5))
            else:
                plt.legend(loc= labels_loc, fancybox=True, prop={'size': 12}, framealpha=0.4)
            plt.show()
    
    def findings_plot(self, prob = None, OG=None, FG=None, KKT=None, PDG = None, SDG=None, OG_bounds=None, 
                        KKT_SDG_bounds=None, PDG_SDG_bounds=None, step=1, min_ite=None, measures=True, show_titles=True, BP_kwargs={}):
        """
        Plots our findings.

        Parameters:
            prob (str): Optimization problem
            OG (numpy.ndarray): Optimality gap.
            FG (numpy.ndarray): Feasibility gap.
            KKT (numpy.ndarray): Karush–Kuhn–Tucker error.
            PDG (numpy.ndarray): Projected duality gap.
            SDG (numpy.ndarray): Smoothed duality gap.
            OG_bounds (dict): Optimality gap bounds.
            KKT_SDG_bounds (dict): KKT and SDG approximation of each other.
            PDG_SDG_bounds (dict): PDG and SDG approximation of each other.
            step (int): The curve values have been appended every 'step' iteration(s).
            min_ite (int): The length of the shorter array when plotting two arrays of different lengths.
            measures (bool): Whether the plot the measures or not.
            show_titles (bool): Whether to show titles for plots.
            BP_kwargs (dict): More arguments for the Basis Pursuit experiment. 
                              For instance, KKT and SDG of version 2 of PDHG. 

        """
        xlabel = 'Iteration/{}'.format(step)
        if prob == 'Basis Pursuit':
            KKT2, SDG2 = BP_kwargs['KKT2'], BP_kwargs['SDG2']
            self.SDG_stability(KKT[:min_ite], KKT2[:min_ite], SDG[:min_ite], SDG2[:min_ite], step=step)
        if measures:
            measures_dict = self.measures_dict(OG, FG, KKT, PDG, SDG)
            self.plot(measures_dict, xlabel=xlabel, show_titles=show_titles)
        if OG_bounds is not None:
            OG_dict = self.OG_dict(OG, FG, OG_bounds)
            self.plot(OG_dict, xlabel=xlabel, show_titles=show_titles)
        if KKT_SDG_bounds is not None:
            KKT_SDG_dict = self.KKT_SDG_dict(KKT, KKT_SDG_bounds)
            self.plot(KKT_SDG_dict, xlabel=xlabel, show_titles=show_titles)
        if PDG_SDG_bounds is not None:
            PDG_SDG_dict = self.PDG_SDG_dict(PDG, PDG_SDG_bounds)
            self.plot(PDG_SDG_dict, xlabel=xlabel, show_titles=show_titles)

    def paper_plots(self, prob, OG=None, FG=None, KKT=None, PDG = None, SDG = None,
                        OG_bounds=None, KKT_SDG_bounds=None, PDG_SDG_bounds=None, step=1, min_ite=None, show_titles=False, BP_kwargs={}):
        """
        Plots the paper's figures.

        Parameters:
            prob (str): Optimization problem
            OG (numpy.ndarray): Optimality gap.
            FG (numpy.ndarray): Feasibility gap.
            KKT (numpy.ndarray): Karush–Kuhn–Tucker error.
            PDG (numpy.ndarray): Projected duality gap.
            SDG (numpy.ndarray): Smoothed duality gap.
            OG_bounds (dict): Optimality gap bounds.
            KKT_SDG_bounds (dict): KKT and SDG approximation of each other.
            PDG_SDG_bounds (dict): PDG and SDG approximation of each other.
            step (int): The curve values have been appended every 'step' iteration(s).
            min_ite (int): The length of the shorter array when plotting two arrays of different lengths.
            show_titles (bool): Whether to show titles for plots.
            BP_kwargs (dict): More arguments for the Basis Pursuit experiment. 
                              For instance, KKT and SDG of version 2 of PDHG. 
        """
        one_dim_dict, IID_dict, cov_dict, distributed_dict, QP_dict, BP_dict = self.paper_plots_dicts(OG, 
                                                        FG, KKT, PDG, OG_bounds, KKT_SDG_bounds, PDG_SDG_bounds)
        xlabel = 'Iteration/{}'.format(step)
        if prob == 'One-dimensional':
            self.plot(one_dim_dict, xlabel=xlabel, show_titles=show_titles)
        elif prob == 'I.I.D. Gaussian matrices':
            self.plot(IID_dict, xlabel=xlabel, show_titles=show_titles)
        elif prob == 'Non-trivial covariance':
            self.plot(cov_dict, xlabel=xlabel, show_titles=show_titles)
        elif prob == 'Distributed optimization':
            self.plot(distributed_dict, xlabel=xlabel, show_titles=show_titles)
        elif prob == 'Quadratic programming':
            self.plot(QP_dict, xlabel=xlabel, show_titles=show_titles)
        elif prob == 'Basis Pursuit':
            KKT2, SDG2 = BP_kwargs['KKT2'], BP_kwargs['SDG2']
            self.SDG_stability(KKT[:min_ite], KKT2[:min_ite], SDG[:min_ite], SDG2[:min_ite], step=step, show_titles=show_titles)
            self.plot(BP_dict, xlabel=xlabel, show_titles=show_titles)
        else:
            raise TypeError("""The specified problem is not correct, please choose one of the following:
                                * One-dimensional
                                * I.I.D. Gaussian matrices
                                * Non-trivial covariance
                                * Distributed optimization
                                * Quadratic programming
                                * Basis Pursuit
                            """)
    
    def SDG_stability(self, KKT, KKT2, SDG, SDG2, step = 1, show_titles=True):
        """
        Plots version 1 vs. version 2 of PDHG for both: the KKT error and SDG

        Parameters:
            KKT (numpy.ndarray): Karush–Kuhn–Tucker error computed at the solution of version 1.
            KKT2 (numpy.ndarray): Karush–Kuhn–Tucker error computed at the solution of version 2.
            SDG (numpy.ndarray): Smoothed duality gap computed at the solution of version 1.
            SDG2 (numpy.ndarray): Smoothed duality gap computed at the solution of version 2.
            step (int): The curve values have been appended every 'step' iteration(s).
            show_titles (bool): Whether to show titles for plots.
        """
        xlabel = 'Iteration/{}'.format(step)
        fig, (ax1, ax2) = plt.subplots(1, 2)  # Increase figure size to (width, height)
        ax1.semilogy(KKT, label="PDHG Version 1")
        ax1.semilogy(KKT2, label='PDHG Version 2')
        ax2.semilogy(SDG, label="PDHG Version 1")
        ax2.semilogy(SDG2, label="PDHG Version 2")

        # Adding grid and labels for ax1
        ax1.grid(True)  # Add grid to the first subplot
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel('Karush–Kuhn–Tucker error')
        ax1.legend(loc='best', fancybox=True, prop={'size': 10}, framealpha=0.3)

        # Adding grid and labels for ax2
        ax2.grid(True)  # Add grid to the second subplot
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel('Smoothed Duality Gap')
        ax2.legend(loc='lower left', fancybox=True, prop={'size': 10}, framealpha=0.3)

        # Twin the y-axis of ax2 to create a single y-axis on the right side
        ax2.yaxis.tick_right()  # Move the y-axis ticks to the right side
        ax2.yaxis.set_label_position("right")  # Set the label position to the right side

        if show_titles:
            plt.suptitle("Superior stability of SDG over the KKT error")  # Adding the main title
        plt.tight_layout()  # Adjusts subplot parameters to fit the figure area
        plt.show()

    def plot_all(self, paper=True, prob=None, OG=None, FG=None, KKT=None, PDG = None, SDG = None, OG_bounds=None, 
                 KKT_SDG_bounds=None, PDG_SDG_bounds=None, step=1, min_ite=None, measures=True, show_titles=True, BP_kwargs={}):
        """
        Plots either the paper's figures or all the plots of our findings.

        Parameters:
            paper (bool): Whether to plot the paper's plots or our findings plots. 
            prob (str): Optimization problem
            OG (numpy.ndarray): Optimality gap.
            FG (numpy.ndarray): Feasibility gap.
            KKT (numpy.ndarray): Karush–Kuhn–Tucker error.
            PDG (numpy.ndarray): Projected duality gap.
            SDG (numpy.ndarray): Smoothed duality gap.
            OG_bounds (dict): Optimality gap bounds.
            KKT_SDG_bounds (dict): KKT and SDG approximation of each other.
            PDG_SDG_bounds (dict): PDG and SDG approximation of each other.
            step (int): The curve values have been appended every 'step' iteration(s).
            min_ite (int): The length of the shorter array when plotting two arrays of different lengths.
            measures (bool): Whether the plot the measures or not.
            show_titles (bool): Whether to show titles for plots.
            BP_kwargs (dict): More arguments for the Basis Pursuit experiment. 
                              For instance, KKT and SDG of version 2 of PDHG. 
        """
        if paper:
            self.paper_plots(prob, OG, FG, KKT, PDG, SDG, OG_bounds, KKT_SDG_bounds, PDG_SDG_bounds, step, min_ite, show_titles, BP_kwargs)
        else:
            self.findings_plot(prob, OG, FG, KKT, PDG, SDG, OG_bounds, KKT_SDG_bounds, PDG_SDG_bounds, step, min_ite, measures, show_titles, BP_kwargs) 

            
        