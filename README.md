# fasttomo
<!-- Start with a clear and concise title that reflects the purpose of your project. Follow it with a brief description that outlines what your pipeline does. -->

Welcome to **`fasttomo`**, a Python package designed to address the critical task of identifying copper agglomerates in lithium-ion batteries during thermal runaway. Developed as part of my Master's thesis at ESRF (European Synchrotron Radiation Facility) from October 2023 to March 2024, `fasttomo` offers a powerful set of tools for analyzing and visualizing 4-dimensional CT data in $(t, z, y, x)$ format.

**Key Features:**

- Load and process 4-dimensional CT data stored as `NumPy` arrays (`.npy` files).
- Perform consistent segmentation of copper agglomerates over time.
- Visualize data interactively, with the option to overlay segmentation masks.
- Extract and plot various agglomerate features, such as volume, speed, and density.
- Generate `.stl` mesh representations of the segmentation masks.
- Seamlessly import `.stl` meshes into Blender for detailed rendering.
- Create dynamic movies by sequencing still images.

`fasttomo` is a valuable tool for researchers, scientists, and engineers working on battery technology, providing insights into the behavior and characteristics of copper agglomerates in lithium-ion batteries under thermal stress.

For detailed instructions on installing, configuring, and using the package, please refer to the comprehensive [documentation](https://fasttomo.readthedocs.io).

## 1. Installation
<!-- Provide detailed instructions on how to install your pipeline. Include any dependencies and system requirements. You might also want to include installation commands for easy setup. -->
To install `fasttomo`, follow these steps:

1. First, clone the repository to your local machine:

   ```bash
   git clone https://github.com/venturellimatteo/fasttomo.git
  
2. Navigate to the project directory:

   ```bash
   cd fasttomo
  
3. Create a conda virtual environment:

   ```bash
   conda create -n venv

4. Activate the virtual environment:

   ```bash
   conda activate venv

5. Install `Python 3.8.17`:

   ```bash
   conda install python==3.8.17

6. Install `requirements.txt` dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## 2. Usage
<!-- Clearly explain how to use your pipeline. Include examples and command-line syntax if applicable. If there are configuration files, provide information on how to customize them. -->

## 3. Examples
<!-- Include examples of how to use your pipeline with sample inputs. This helps users understand the expected input format and see the output. -->

## 4. Results
<!-- If your pipeline produces visual results, consider including sample outputs or screenshots to showcase the expected outcomes. -->

## 5. Documentation
<!-- If you have detailed documentation beyond the README, provide links to it. This could include API documentation, user guides, or tutorials. -->
For detailed documentation, including API references and usage guides, please visit the [project documentation](https://fasttomo.readthedocs.io).

If you encounter any issues, have questions, or would like to suggest improvements to the documentation, please [open an issue](https://github.com/venturellimatteo/fasttomo/issues) on GitHub.

## 6. License
<!-- Specify the license under which your project is released. This is important for users who want to understand how they can use, modify, and distribute your code. -->

This project is licensed under the [MIT License](LICENSE.md) - see the [LICENSE.md](LICENSE.md) file for details.

The MIT License is a permissive open-source license that allows you to use, modify, and distribute the code in both open-source and proprietary projects. For more details, please refer to the [MIT License](https://opensource.org/licenses/MIT).

## 7. Acknowledgments
<!-- Give credit to any external libraries, tools, or resources that you used in your project. This is a good practice to show appreciation for the work of others. -->

<!-- I would like to extend my heartfelt gratitude to two remarkable individuals who played pivotal roles in the success of this project, not only as supervisors but also as friends.

**Matilda** has been an extraordinary mentor, friend, and source of inspiration. Beyond her exceptional guidance in the academic realm, Matilda shared countless precious moments and provided unwavering support.

**Ludo** is not just a mentor but a remarkable professional with an infectious smile. His cheerful attitude and deep professional expertise have been a constant source of motivation. Ludo's valuable advice and positive outlook have consistently guided me through challenges, making the journey smoother and more enjoyable.

I feel profoundly fortunate to have Matilda and Ludo as supervisors and friends, and their influence on this project and my personal development cannot be overstated.

Thank you for being such essential pillars of support and belief in this project. -->

## 8. Contact Information
<!-- Provide a way for users to contact you if they have questions, feedback, or want to collaborate. This could be an email address, a link to your personal website, or a discussion forum. -->
If you have any questions, feedback, or suggestions, feel free to reach out:

- **Email**: <matteo.venturelli2000@gmail.com>

You can also open an issue on GitHub or participate in discussions on my [project's GitHub repository](https://github.com/VenturelliMatteo/MasterThesis).

Feel free to connect with me on [LinkedIn](https://www.linkedin.com/in/matteo-venturelli/) for more updates and collaboration opportunities.
