# PCanalyzer
PCA analyzer Characterizing conformational dynamic via PCA   
# File upoad
Drag and drop the multiple pdb files 
<img width="954" height="438" alt="image" src="https://github.com/user-attachments/assets/603ff81a-2be8-4859-8aee-6dc2a8096bb2" /> 
## 📊 PCA Projection (3D or 2D)

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

<img width="432" height="326" alt="image" src="https://github.com/user-attachments/assets/b81c3e86-de1a-406b-8357-dd419663fd4b" />
