import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from data_utils import load_pdb_atom_and_hetatm_coordinates_from_file, load_pdb_coordinates_from_file
from rmsd_utils import calculate_rmsd_matrix

# --- Streamlit App ---

st.title("🧬 Principal Component Analyzer")

st.sidebar.header("Upload your PDB files")
uploaded_files = st.sidebar.file_uploader(
    "Choose PDB files",
    type=['pdb'],
    accept_multiple_files=True
)

show_labels = st.sidebar.checkbox("Show labels", value=True)

if uploaded_files:
    # Step 1: Load and prepare the data
    all_data = []
    filenames = []

    for file in uploaded_files:
        coords = load_pdb_coordinates_from_file(file)
        coords_flattened = coords.flatten()
        all_data.append(coords_flattened)
        filenames.append(file.name)

    all_data = np.array(all_data)

    # Step 2: Apply PCA
    n_files = len(uploaded_files)
    n_components = 3
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(all_data)
    explained_var = pca.explained_variance_ratio_

    # Step 3: Map files to PCs
    file_to_pc = np.argmax(np.abs(pca_result), axis=1)

    # st.subheader("📈 File to Principal Component Mapping")
    # for idx, pc in enumerate(file_to_pc):
    #     st.write(f"{filenames[idx]}** --> PC{pc+1}")

    # Step 4: PCA 3D Plot
    st.subheader("📊 PCA Projection (3D or 2D)")

    if pca_result.shape[1] >= 3:
        fig = px.scatter_3d(
            x=pca_result[:, 0], y=pca_result[:, 1], z=pca_result[:, 2],
            color=filenames,
            labels={
                'x': f'PC1 ({explained_var[0]*100:.2f}% var)',
                'y': f'PC2 ({explained_var[1]*100:.2f}% var)',
                'z': f'PC3 ({explained_var[2]*100:.2f}% var)'
            },
            hover_name=filenames
        )
    else:
        fig = px.scatter(
            x=pca_result[:, 0], y=pca_result[:, 1],
            color=filenames,
            labels={
                'x': f'PC1 ({explained_var[0]*100:.2f}% var)',
                'y': f'PC2 ({explained_var[1]*100:.2f}% var)'
            },
            hover_name=filenames
        )

    if not show_labels:
        fig.update_traces(marker=dict(size=8), text=None, hoverinfo='none')

    fig.update_layout(
        width=800,
        height=600,
        title="PCA of Atom Positions",
        legend_title="Files"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Step 5: Explained variance text
    st.subheader("🔍 Explained Variance by Principal Components")
    for idx, var in enumerate(explained_var):
        st.write(f"PC{idx+1}: {var*100:.2f}%")

    # Step 6: 2D PCA projections
    st.subheader("🔎 2D PCA Projections Between Components")

    if n_components >= 2:
        fig_pc1_pc2 = px.scatter(
            x=pca_result[:, 0], y=pca_result[:, 1],
            color=filenames,
            labels={
                'x': f'PC1 ({explained_var[0]*100:.2f}% var)',
                'y': f'PC2 ({explained_var[1]*100:.2f}% var)'
            },
            title="PC1 vs PC2",
            hover_name=filenames
        )
        st.plotly_chart(fig_pc1_pc2, use_container_width=True)

    if n_components == 3:
        fig_pc2_pc3 = px.scatter(
            x=pca_result[:, 1], y=pca_result[:, 2],
            color=filenames,
            labels={
                'x': f'PC2 ({explained_var[1]*100:.2f}% var)',
                'y': f'PC3 ({explained_var[2]*100:.2f}% var)'
            },
            title="PC2 vs PC3",
            hover_name=filenames
        )
        st.plotly_chart(fig_pc2_pc3, use_container_width=True)

        fig_pc1_pc3 = px.scatter(
            x=pca_result[:, 0], y=pca_result[:, 2],
            color=filenames,
            labels={
                'x': f'PC1 ({explained_var[0]*100:.2f}% var)',
                'y': f'PC3 ({explained_var[2]*100:.2f}% var)'
            },
            title="PC1 vs PC3",
            hover_name=filenames
        )
        st.plotly_chart(fig_pc1_pc3, use_container_width=True)

    # Step 7: Explained Variance Bar Chart
    st.subheader("📊 Explained Variance Bar Chart")

    variance_data = {
        'Principal Component': [f'PC{i+1}' for i in range(len(explained_var))],
        'Explained Variance (%)': explained_var * 100
    }

    fig_var = px.bar(
        variance_data,
        x='Principal Component',
        y='Explained Variance (%)',
        text_auto='.2f',
        title="Explained Variance by Principal Components",
        color='Principal Component'
    )

    fig_var.update_layout(
        width=800,
        height=500,
        yaxis_title='Explained Variance (%)',
        xaxis_title='Principal Component',
        showlegend=False
    )

    st.plotly_chart(fig_var, use_container_width=True)

    # Step 8: Clustering
    st.subheader("🔬 KMeans Clustering in PCA Space")

    n_clusters = min(4, len(filenames))  # Avoid more clusters than samples
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    clusters = kmeans.fit_predict(pca_result)

    fig_cluster_3d = px.scatter_3d(
        x=pca_result[:, 0], y=pca_result[:, 1], z=pca_result[:, 2],
        color=clusters.astype(str),
        labels={
            'x': f'PC1 ({explained_var[0]*100:.2f}% var)',
            'y': f'PC2 ({explained_var[1]*100:.2f}% var)',
            'z': f'PC3 ({explained_var[2]*100:.2f}% var)'
        },
        hover_name=filenames,
        title="Clusters in 3D PCA Space"
    )
    st.plotly_chart(fig_cluster_3d, use_container_width=True)

    # Step 9: RMSD Calculation
    st.subheader("📐 RMSD from Reference Structure")

    rmsd_values = calculate_rmsd_matrix(all_data)

    st.subheader("📏 RMSD per File (Bar Plot)")

    fig_rmsd_bar = px.bar(
        x=filenames,
        y=rmsd_values,
        labels={'x': 'PDB File', 'y': 'RMSD'},
        title="Individual RMSD per File",
        text=[f"{r:.3f}" for r in rmsd_values]
    )
    fig_rmsd_bar.add_shape(
        type="line",
        x0=-0.5,
        x1=len(filenames) - 0.5,
        y0=rmsd_values[0],
        y1=rmsd_values[0],
        line=dict(color="Red", width=2, dash="dash")
    )
    fig_rmsd_bar.update_layout(
        width=800,
        height=500
    )
    st.plotly_chart(fig_rmsd_bar, use_container_width=True)

    # Step 10: RMSD Comparison Between ATOM and HETATM
    st.subheader("🧪 RMSD Comparison: ATOM vs HETATM")

    atom_coords_list = []
    hetatm_coords_list = []

    for file in uploaded_files:
        file.seek(0)
        atom_coords, hetatm_coords = load_pdb_atom_and_hetatm_coordinates_from_file(file)
        atom_coords_list.append(atom_coords.flatten())
        hetatm_coords_list.append(hetatm_coords.flatten())

    all_data_atom = np.array(atom_coords_list)
    all_data_hetatm = np.array(hetatm_coords_list)

    rmsd_atom = calculate_rmsd_matrix(all_data_atom)
    rmsd_hetatm = calculate_rmsd_matrix(all_data_hetatm)

    # ATOM bar plot
    fig_rmsd_atom_bar = px.bar(
        x=filenames,
        y=rmsd_atom,
        labels={'x': 'File', 'y': 'RMSD'},
        title="ATOM RMSD (Bar Plot)",
        text=[f"{r:.3f}" for r in rmsd_atom]
    )
    fig_rmsd_atom_bar.add_shape(
        type="line",
        x0=-0.5,
        x1=len(filenames) - 0.5,
        y0=rmsd_atom[0],
        y1=rmsd_atom[0],
        line=dict(color="Red", width=2, dash="dash")
    )
    fig_rmsd_atom_bar.update_layout(width=800, height=500)
    st.plotly_chart(fig_rmsd_atom_bar, use_container_width=True)

    # HETATM bar plot
    fig_rmsd_hetatm_bar = px.bar(
        x=filenames,
        y=rmsd_hetatm,
        labels={'x': 'File', 'y': 'RMSD'},
        title="HETATM RMSD (Bar Plot)",
        text=[f"{r:.3f}" for r in rmsd_hetatm]
    )
    fig_rmsd_hetatm_bar.add_shape(
        type="line",
        x0=-0.5,
        x1=len(filenames) - 0.5,
        y0=rmsd_hetatm[0],
        y1=rmsd_hetatm[0],
        line=dict(color="Red", width=2, dash="dash")
    )
    fig_rmsd_hetatm_bar.update_layout(width=800, height=500)
    st.plotly_chart(fig_rmsd_hetatm_bar, use_container_width=True)

else:
    st.info("⬆ Upload one or more PDB files to begin.")
