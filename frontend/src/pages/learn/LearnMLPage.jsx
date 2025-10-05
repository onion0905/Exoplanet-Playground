import React from "react";
import { useNavigate } from "react-router-dom";
import Navbar from "../../components/Navbar";
import CSSBackgroundVisualization from "../../components/CSSBackgroundVisualization";
import SimpleTrailCursor from "../../components/SimpleTrailCursor";
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import { Button } from '@mui/material';

function LearnMLPage() {
  const navigate = useNavigate();

  return (
    <div className="relative w-full min-h-screen">
      <CSSBackgroundVisualization />
      <SimpleTrailCursor />
      <Navbar />

      <main className="relative z-10 px-8 pt-32 max-w-6xl mx-auto">
        <h1 className="font-bold text-white text-5xl mb-10">
          Learn about machine learning
        </h1>

        {/* Flex container for main content and right-side photo placeholder */}
        <div className="flex flex-col lg:flex-row gap-8">
          <div className="flex-grow">
            {/* Brief introduction of machine learning */}
            <section className="mb-12">
              <h2 className="font-bold text-white text-3xl mb-6">Brief introduction of machine learning</h2>
              <div className="bg-transparent backdrop-blur-sm border border-gray-600/30 rounded-2xl p-6 shadow-2xl">
                <p className="text-white text-lg leading-relaxed mb-4">
                  Artificial Intelligence (AI) refers to any system or software capable of learning, reasoning, or
                  performing tasks in ways that appear intelligent. Machine Learning (ML) is a key branch of AI
                  focused on enabling computers to learn from data without explicit programming. In essence, ML
                  allows algorithms to "self-improve" – discovering patterns and relationships automatically. Even
                  simple techniques like linear regression or least-squares fitting are early forms of machine
                  learning, as they enable a model to adapt based on data.
          </p>
        </div>
            </section>

            {/* Introduction of the model we used */}
            <section className="mb-12">
              <h2 className="font-bold text-white text-3xl mb-6">Introduction of the model we used</h2>

              {/* 1. Decision Tree */}
              <div className="mb-8 bg-transparent backdrop-blur-sm border border-gray-600/30 rounded-2xl p-6 shadow-2xl">
                <div className="flex items-start gap-6">
                  <div className="flex-grow">
                    <h3 className="font-bold text-white text-2xl mb-3">1. Decision Tree</h3>
                    <p className="text-white text-lg leading-relaxed mb-4">
                      Overview: A tree-based model that recursively splits data into smaller
                      subsets to make predictions.
                    </p>
                    <p className="font-semibold text-white text-xl mb-3">Key Hyperparameters:</p>
                    <ul className="list-disc list-inside ml-4 text-gray-300 space-y-2">
                      <li>
                        <span className="font-mono text-blue-300 bg-blue-900/30 px-2 py-1 rounded">max_depth</span> → limits how deep the tree can grow; prevents overfitting.
                      </li>
                      <li>
                        <span className="font-mono text-blue-300 bg-blue-900/30 px-2 py-1 rounded">min_samples_split</span> → minimum samples required to split a node;
                        higher value → simpler model.
                      </li>
                      <li>
                        <span className="font-mono text-blue-300 bg-blue-900/30 px-2 py-1 rounded">min_samples_leaf</span> → minimum samples required at a leaf node;
                        smooths the model.
                      </li>
                      <li>
                        <span className="font-mono text-blue-300 bg-blue-900/30 px-2 py-1 rounded">criterion</span> → measures split quality (gini or entropy for classification,
                        squared_error for regression).
                      </li>
                    </ul>
                  </div>
                  <div className="flex-shrink-0">
                    <img src="/learnML/DT.png" alt="Decision Tree" className="w-auto h-32" />
                  </div>
                </div>
              </div>

              {/* 2. Random Forest */}
              <div className="mb-8 bg-transparent backdrop-blur-sm border border-gray-600/30 rounded-2xl p-6 shadow-2xl">
                <div className="flex items-start gap-6">
                  <div className="flex-grow">
                    <h3 className="font-bold text-white text-2xl mb-3">2. Random Forest</h3>
                    <p className="text-white text-lg leading-relaxed mb-4">
                      Overview: An ensemble of many decision trees trained on random
                      subsets of data and features.
                    </p>
                    <p className="font-semibold text-white text-xl mb-3">Key Hyperparameters:</p>
                    <ul className="list-disc list-inside ml-4 text-gray-300 space-y-2">
                      <li>
                        <span className="font-mono text-blue-300 bg-blue-900/30 px-2 py-1 rounded">n_estimators</span> → number of trees; more trees improve accuracy but
                        increase compute time.
                      </li>
                      <li>
                        <span className="font-mono text-blue-300 bg-blue-900/30 px-2 py-1 rounded">max_depth</span> → limits tree depth; prevents overfitting.
                      </li>
                      <li>
                        <span className="font-mono text-blue-300 bg-blue-900/30 px-2 py-1 rounded">max_features</span> → number of features considered at each split; controls
                        randomness and diversity.
                      </li>
                      <li>
                        <span className="font-mono text-blue-300 bg-blue-900/30 px-2 py-1 rounded">min_samples_leaf</span> → minimum number of samples in each leaf;
                        reduces overfitting.
                      </li>
                    </ul>
                  </div>
                  <div className="flex-shrink-0">
                    <img src="/learnML/RF.png" alt="Random Forest" className="w-auto h-32" />
                  </div>
                </div>
              </div>

              {/* 3. Linear Regression */}
              <div className="mb-8 bg-transparent backdrop-blur-sm border border-gray-600/30 rounded-2xl p-6 shadow-2xl">
                <div className="flex items-start gap-6">
                  <div className="flex-grow">
                    <h3 className="font-bold text-white text-2xl mb-3">3. Linear Regression</h3>
                    <p className="text-white text-lg leading-relaxed mb-4">
                      Overview: Models the relationship between inputs and a continuous
                      target using a linear equation.
                    </p>
                    <p className="font-semibold text-white text-xl mb-3">Key Hyperparameters:</p>
                    <ul className="list-disc list-inside ml-4 text-gray-300 space-y-2">
                      <li>
                        <span className="font-mono text-blue-300 bg-blue-900/30 px-2 py-1 rounded">fit_intercept</span> → whether to include a bias term; usually True.
                      </li>
                      <li>
                        <span className="font-mono text-blue-300 bg-blue-900/30 px-2 py-1 rounded">normalize</span> → whether to normalize input features before fitting.
                      </li>
                    </ul>
                  </div>
                  <div className="flex-shrink-0">
                    <img src="/learnML/LR.png" alt="Linear Regression" className="w-auto h-32" />
                  </div>
                </div>
              </div>

              {/* 4. Principal Component Analysis (PCA) */}
              <div className="mb-8 bg-transparent backdrop-blur-sm border border-gray-600/30 rounded-2xl p-6 shadow-2xl">
                <div className="flex items-start gap-6">
                  <div className="flex-grow">
                    <h3 className="font-bold text-white text-2xl mb-3">4. Principal Component Analysis (PCA)</h3>
                    <p className="text-white text-lg leading-relaxed mb-4">
                      Overview: Reduces dimensionality by projecting data onto orthogonal
                      axes that capture maximum variance.
                    </p>
                    <p className="font-semibold text-white text-xl mb-3">Key Hyperparameters:</p>
                    <ul className="list-disc list-inside ml-4 text-gray-300 space-y-2">
                      <li>
                        <span className="font-mono text-blue-300 bg-blue-900/30 px-2 py-1 rounded">n_components</span> → number of principal components to keep; fewer
                        components = more compression.
                      </li>
                      <li>
                        <span className="font-mono text-blue-300 bg-blue-900/30 px-2 py-1 rounded">svd_solver</span> → method for decomposition (auto, full, randomized,
                        arpack).
                      </li>
                    </ul>
                  </div>
                  <div className="flex-shrink-0">
                    <img src="/learnML/PCA.png" alt="PCA" className="w-auto h-32" />
                  </div>
                </div>
              </div>

              {/* 5. Support Vector Machine (SVM) */}
              <div className="mb-8 bg-transparent backdrop-blur-sm border border-gray-600/30 rounded-2xl p-6 shadow-2xl">
                <div className="flex items-start gap-6">
                  <div className="flex-grow">
                    <h3 className="font-bold text-white text-2xl mb-3">5. Support Vector Machine (SVM)</h3>
                    <p className="text-white text-lg leading-relaxed mb-4">
                      Overview: Finds an optimal hyperplane that separates data points of
                      different classes.
                    </p>
                    <p className="font-semibold text-white text-xl mb-3">Key Hyperparameters:</p>
                    <ul className="list-disc list-inside ml-4 text-gray-300 space-y-2">
                      <li>
                        <span className="font-mono text-blue-300 bg-blue-900/30 px-2 py-1 rounded">C</span> → regularization strength; smaller = smoother boundary (less
                        overfitting).
                      </li>
                      <li>
                        <span className="font-mono text-blue-300 bg-blue-900/30 px-2 py-1 rounded">kernel</span> → transformation of input space (linear, poly, rbf, sigmoid).
                      </li>
                      <li>
                        <span className="font-mono text-blue-300 bg-blue-900/30 px-2 py-1 rounded">gamma</span> → controls influence of individual samples in rbf/poly
                        kernels; higher = more complex model.
                      </li>
                    </ul>
                  </div>
                  <div className="flex-shrink-0">
                    <img src="/learnML/SVM.png" alt="SVM" className="w-auto h-32" />
                  </div>
                </div>
              </div>

              {/* 6. Neural Network */}
              <div className="mb-8 bg-transparent backdrop-blur-sm border border-gray-600/30 rounded-2xl p-6 shadow-2xl">
                <div className="flex items-start gap-6">
                  <div className="flex-grow">
                    <h3 className="font-bold text-white text-2xl mb-3">6. Neural Network</h3>
                    <p className="text-white text-lg leading-relaxed mb-4">
                      Overview: A layered network that learns complex nonlinear relationships
                      between input and output.
                    </p>
                    <p className="font-semibold text-white text-xl mb-3">Key Hyperparameters:</p>
                    <ul className="list-disc list-inside ml-4 text-gray-300 space-y-2">
                      <li>
                        <span className="font-mono text-blue-300 bg-blue-900/30 px-2 py-1 rounded">hidden_layers</span> → number and size of hidden layers; deeper/wider =
                        more capacity, risk of overfitting.
                      </li>
                      <li>
                        <span className="font-mono text-blue-300 bg-blue-900/30 px-2 py-1 rounded">activation</span> → nonlinear transformation (relu, sigmoid, tanh, softmax).
                      </li>
                      <li>
                        <span className="font-mono text-blue-300 bg-blue-900/30 px-2 py-1 rounded">learning_rate</span> → how much weights change during training; too high =
                        unstable, too low = slow learning.
                      </li>
                    </ul>
                  </div>
                  <div className="flex-shrink-0">
                    <img src="/learnML/NN.png" alt="Neural Network" className="w-auto h-32" />
                  </div>
                </div>
              </div>
            </section>

            {/* AI Astrobiology Exoplanetary Science Literature Review */}
            <section className="mb-12">
              <h2 className="font-bold text-white text-3xl mb-6">AI Astrobiology Exoplanetary Science Literature Review</h2>
              
              <div className="bg-transparent backdrop-blur-sm border border-gray-600/30 rounded-2xl p-6 shadow-2xl">
                <div className="mb-6">
                  <h3 className="font-bold text-white text-2xl mb-3">2023:</h3>
                  <ul className="list-lisc list-inside ml-4 text-gray-300 space-y-4">
                    <li>
                      <p className="text-white">
                        Identifying Exoplanets with Deep Learning. IV. Removing Stellar Activity Signals from Radial
                        Velocity Measurements Using Neural Networks. 2022,de Beurs, Z. L., et al.<br />
                        <a href="https://iopscience.iop.org/article/10.3847/1538-3881/ac738e/meta" target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:underline">https://iopscience.iop.org/article/10.3847/1538-3881/ac738e/meta</a><br />
                        <span className="text-blue-300">Keywords: Exoplanets, Deep Learning, Feature Modeling</span>
                      </p>
                    </li>
                    <li>
                      <p className="text-white">
                        Exoplanet atmosphere evolution: emulation with neural networks. 2023, Rogers, J. G. et
                        al. Monthly Notices of the Royal Astronomical Society. <br />
                        <a href="https://doi.org/10.1093/mnras/stad089" target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:underline">https://doi.org/10.1093/mnras/stad089</a><br />
                        <span className="text-blue-300">Keywords: Exoplanets, Neural Nets, Inference, Emulation</span>
                      </p>
                    </li>
                    <li>
                      <p className="text-white">
                        Multiplicity Boost of Transit Signal Classifiers: Validation of 69 New Exoplanets using the
                        Multiplicity Boost of ExoMiner. 2023, Valizadegan, H. et al. The Astronomical Journal.<br />
                        <a href="https://iopscience.iop.org/article/10.3847/1538-3881/acd344" target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:underline">https://iopscience.iop.org/article/10.3847/1538-3881/acd344</a><br />
                        <span className="text-blue-300">Keywords: Exoplanets, Classification, Deep Learning</span>
                      </p>
                    </li>
                  </ul>
                </div>

                <div className="mb-6">
                  <h3 className="font-bold text-white text-2xl mb-3">2022:</h3>
                  <ul className="list-lisc list-inside ml-4 text-gray-300 space-y-4">
                    <li>
                      <p className="text-white">
                        ExoMiner: A Highly Accurate and Explainable Deep Learning Classifier That Validates 301 New
                        Exoplanets. 2022, Valizadegan, H., The Astrophysical Journal.<br />
                        <a href="https://iopscience.iop.org/article/10.3847/1538-4357/ac4399/meta" target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:underline">https://iopscience.iop.org/article/10.3847/1538-4357/ac4399/meta</a><br />
                        <span className="text-blue-300">Keywords: Exoplanets, Deep Learning, Classification</span>
                      </p>
                    </li>
                  </ul>
                </div>

                <div className="mb-6">
                  <h3 className="font-bold text-white text-2xl mb-3">2021:</h3>
                  <ul className="list-lisc list-inside ml-4 text-gray-300 space-y-4">
                    <li>
                      <p className="text-white">
                        Exoplanet detection using machine learning. 2021, Malik, A., Monthly Notices of the Royal
                        Astronomical Society.<br />
                        <a href="https://doi.org/10.1093/mnras/stab3692" target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:underline">https://doi.org/10.1093/mnras/stab3692</a><br />
                        <span className="text-blue-300">Keywords: Photometry, Classifiers, Gradient Boosted Tree, Exoplanets</span>
                      </p>
                    </li>
                  </ul>
                </div>

                <div className="mb-6">
                  <h3 className="font-bold text-white text-2xl mb-3">2018:</h3>
                  <ul className="list-lisc list-inside ml-4 text-gray-300 space-y-4">
                    <li>
                      <p className="text-white">
                        Supervised machine learning for analysing spectra of exoplanetary
                        atmospheres. 2018, Márquez-Neila, P. et al. Nat Astron<br />
                        <a href="https://doi.org/10.1038/s41550-018-0504-2" target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:underline">https://doi.org/10.1038/s41550-018-0504-2</a><br />
                        <span className="text-blue-300">Keywords: Exoplanets, Classification, Random Forest</span>
                      </p>
                    </li>
                  </ul>
                </div>
              </div>
            </section>

            {/* Reference */}
            <section className="mb-12">
              <h2 className="font-bold text-white text-3xl mb-6">Reference:</h2>
              <div className="bg-transparent backdrop-blur-sm border border-gray-600/30 rounded-2xl p-6 shadow-2xl">
                <ul className="list-disc list-inside ml-4 text-gray-300 space-y-2">
                  <li>
                    <a href="https://www.nasa.gov/space-science-and-astrobiology-at-ames/research-teams/ai-astrobiology/" target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:underline">
                      [1] https://www.nasa.gov/space-science-and-astrobiology-at-ames/research-teams/ai-astrobiology/
                    </a>
                  </li>
                  <li>
                    <a href="https://www.csie.ntu.edu.tw/~htlin/course/ml24fall/" target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:underline">
                      [2] https://www.csie.ntu.edu.tw/~htlin/course/ml24fall/
                    </a>
                  </li>
                </ul>
              </div>
            </section>
          </div>

        </div>
      </main>
    </div>
  );
}

export default LearnMLPage;
