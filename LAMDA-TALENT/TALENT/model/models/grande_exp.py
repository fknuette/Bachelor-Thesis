import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import category_encoders as ce
import sklearn

class GRANDE(nn.Module):
    def __init__(self, *, batch_size, num_objectives, task_type, **args):
        super(GRANDE, self).__init__()
        self.set_params(**args)
        
        self.batch_size = batch_size
        self.num_objectives = num_objectives
        self.task_type = task_type

        self.internal_node_num_ = 2 ** self.depth - 1
        self.leaf_node_num_ = 2 ** self.depth
        
    def forward(self, inputs):
        inputs = inputs.to(dtype=torch.float16)
        training = self.training

        if self.data_subset_fraction < 1.0 and training:
            X_estimator = inputs[:, self.features_by_estimator]
            X_estimator = X_estimator[self.data_select]
        else:
            X_estimator = inputs[:, self.features_by_estimator]

        
        # Iterate entMax through the objectives
        splits = torch.chunk(self.split_index_array, chunks=self.num_objectives, dim=-1) 
        entmax_outputs = []
        for part in splits:
            part_squeezed = part.squeeze(-1)  # Nur die letzte Dimension entfernen (z. B. (B, 1) → (B,))
            entmax_result = self.entmax15(part_squeezed)

            #Here we make the index selecter
            idx = torch.argmax(entmax_result, dim=-1)
            one_hot = nn.functional.one_hot(idx, num_classes=entmax_result.shape[-1]).float()
            entmax_result = entmax_result - (entmax_result - one_hot)

            entmax_result = entmax_result.unsqueeze(-1)  # Form wiederherstellen, z. B. (B,) → (B, 1)
            entmax_outputs.append(entmax_result)
        split_index_array = torch.cat(entmax_outputs, dim=-1)

        
        # import ipdb; ipdb.set_trace()

        #argmax_mask = nn.functional.one_hot(torch.argmax(split_index_array, dim=2, keepdim=True), num_classes=split_index_array.shape[2]).float()
        # split_index_array = split_index_array - (split_index_array - nn.functional.one_hot(torch.argmax(split_index_array, dim=-1), num_classes=split_index_array.shape[-1]).float())
        X_estimator = X_estimator.to(dtype=torch.float16)
        split_index_array = split_index_array.to(inputs.device)
        # print(self.split_values.dtype)
        # print(split_index_array.dtype)
        
        
        s1_sum = torch.einsum("eivc,eivc->eic",  self.split_values.float(),   split_index_array.float()).half()
        s2_sum = torch.einsum("bevc,eivc->beic", X_estimator.unsqueeze(-1).repeat(1, 1, 1, 2).float(), split_index_array.float()).half()
        #s1_sum = torch.einsum("ein,ein->ei", self.split_values.float(), split_index_array.float()).half()
        #s2_sum = torch.einsum("ben,ein->bei", X_estimator.float(), split_index_array.float()).half()

        #look if node is one or zero
        node_result = (torch.nn.functional.softsign(s1_sum.unsqueeze(0) - s2_sum) + 1) / 2
        node_result_corrected = node_result - (node_result - torch.round(node_result))

        node_result_extended = node_result_corrected[:, :, self.internal_node_index_list, :]
        # print(self.path_identifier_list.device)
        # print(node_result_extended.device)
        self.path_identifier_list = self.path_identifier_list.to(inputs.device)
        path_ids    = self.path_identifier_list.unsqueeze(-1)  
        p = torch.prod(((1 - path_ids) * node_result_extended + path_ids * (1 - node_result_extended)), dim=3)

        estimator_weights_leaf = torch.einsum("el,belc->bec", self.estimator_weights, p)
        estimator_weights_leaf_softmax = torch.nn.functional.softmax(estimator_weights_leaf.float(), dim=1).half()

        if self.dropout > 0 and training:
            estimator_weights_leaf_softmax = nn.functional.dropout(estimator_weights_leaf_softmax, p=self.dropout)
            
        if self.task_type == 'regression':
            leaf_output = torch.einsum('elc,belc->bec', self.leaf_classes_array.float(), p.float()).half()
            layer_output = estimator_weights_leaf_softmax * leaf_output 
            # layer_output = torch.einsum('be,bec->bec', estimator_weights_leaf_softmax, layer_output)
        elif self.task_type == 'binclass':
            if self.from_logits:
                layer_output = torch.einsum('el,bel->be', self.leaf_classes_array, p)
            else:
                layer_output = torch.sigmoid(torch.einsum('el,bel->be', self.leaf_classes_array, p))
            layer_output = torch.einsum('be,be->be', estimator_weights_leaf_softmax, layer_output)
        elif self.task_type == 'multiclass':
            if self.from_logits:
                layer_output = torch.einsum('elc,bel->bec', self.leaf_classes_array, p)
                layer_output = torch.einsum('be,bec->bec', estimator_weights_leaf_softmax, layer_output)
            else:
                layer_output = torch.nn.functional.softmax(torch.einsum('elc,bel->bec', self.leaf_classes_array, p), dim=-1)
                layer_output = torch.einsum('be,bec->bec', estimator_weights_leaf_softmax, layer_output)

        if self.data_subset_fraction < 1.0 and training:
            result = torch.zeros(inputs.shape[0], device=inputs.device)
            result[self.data_select] = layer_output.transpose(0, 1)
            result = (result / self.counts) * self.n_estimators
        else:
            if self.task_type == 'regression' or self.task_type == 'binclass':
                if layer_output.ndim == 3:  # shape (batch, n_estimators, num_objectives)
                    result = torch.einsum('bec->bc', layer_output)  # sum over estimators
                else:  # shape (batch, n_estimators), single objective already squeezed
                    result = torch.einsum('be->b', layer_output)
            else:
                result = torch.einsum('bec->bc', layer_output)
        # Zwei Problme, einmal wie ist es mit dem unsqueeze und wie klappt es genau mit dem sum damit nicht dim verschwindet (eine Zeile oben drüber)
        if self.task_type == 'regression':
                if result.ndim == 1:
                    result = result.unsqueeze(1)
        
        #if self.task_type == 'regression' or self.task_type == 'binclass':
            # result = result.unsqueeze(1)
        return result.half()

    def build_model(self):
        if self.selected_variables > 1:
            self.selected_variables = min(self.selected_variables, self.number_of_variables)
        else:
            self.selected_variables = int(self.number_of_variables * self.selected_variables)
            self.selected_variables = min(self.selected_variables, 50)
            self.selected_variables = max(self.selected_variables, 10)
            self.selected_variables = min(self.selected_variables, self.number_of_variables)  
        if self.task_type != 'binclass':
            self.data_subset_fraction = 1.0
        if self.data_subset_fraction < 1.0:
            self.subset_size = int(self.batch_size * self.data_subset_fraction)
            if self.bootstrap:
                self.data_select = torch.randint(0, self.batch_size, (self.n_estimators, self.subset_size))
            else:
                indices = [np.random.choice(self.batch_size, size=self.subset_size, replace=False) for _ in range(self.n_estimators)]
                self.data_select = torch.tensor(indices)

            self.counts = torch.tensor(np.unique(self.data_select.numpy(), return_counts=True)[1], dtype=torch.float64)

        self.features_by_estimator = torch.tensor([np.random.choice(self.number_of_variables, size=self.selected_variables, replace=False) for _ in range(self.n_estimators)])

        self.path_identifier_list = []
        self.internal_node_index_list = []
        for leaf_index in range(self.leaf_node_num_):
            for current_depth in range(1, self.depth + 1):
                path_identifier = (leaf_index // (2 ** (self.depth - current_depth))) % 2
                internal_node_index = 2 ** (current_depth - 1) + leaf_index // (2 ** (self.depth - current_depth + 1)) - 1
                self.path_identifier_list.append(path_identifier)
                self.internal_node_index_list.append(internal_node_index)
        self.path_identifier_list = torch.tensor(self.path_identifier_list).view(-1, self.depth).float()
        self.internal_node_index_list = torch.tensor(self.internal_node_index_list).view(-1, self.depth).int()

        if self.task_type in ['binclass', 'regression']:
            leaf_classes_array_shape = [self.n_estimators, self.leaf_node_num_, self.num_objectives]
        else:  # multiclass
            leaf_classes_array_shape = [self.n_estimators, self.leaf_node_num_, self.number_of_classes]
        # leaf_classes_array_shape = [self.n_estimators, self.leaf_node_num_] if self.task_type in ['binclass', 'regression'] else [self.n_estimators, self.leaf_node_num_, self.number_of_classes]
        weight_shape = [self.n_estimators, self.leaf_node_num_]
        
        # Initialize the shape of the parameters
        self.estimator_weights = nn.Parameter(torch.randn(weight_shape, dtype=torch.float16))

        split_shape = [self.n_estimators, self.internal_node_num_, self.selected_variables, self.num_objectives]
        self.split_values = nn.Parameter(torch.randn(split_shape).to(dtype=torch.float16))
        self.split_index_array = nn.Parameter(torch.randn(split_shape).to(dtype=torch.float16))
        self.leaf_classes_array = nn.Parameter(torch.randn(leaf_classes_array_shape).to(dtype=torch.float16))

    def entmax15(self, inputs, axis=-1):
        def _entmax_inner(inputs):
            inputs = inputs / 2
            inputs -= torch.max(inputs, dim=axis, keepdim=True)[0]
            threshold, _ = self.entmax_threshold_and_support(inputs, axis)
            outputs_sqrt = torch.relu(inputs - threshold)
            outputs = outputs_sqrt ** 2
            return outputs

        return _entmax_inner(inputs)

    def entmax_threshold_and_support(self, inputs, axis=-1):
        num_outcomes = inputs.shape[axis]
        inputs_sorted, _ = torch.sort(inputs, dim=axis, descending=True)
        rho = torch.arange(1, num_outcomes + 1, device=inputs.device, dtype=inputs.dtype)
        mean = torch.cumsum(inputs_sorted, dim=axis) / rho
        mean_sq = torch.cumsum(inputs_sorted ** 2, dim=axis) / rho
        delta = (1 - rho * (mean_sq - mean ** 2)) / rho
        delta_nz = torch.relu(delta)
        tau = mean - torch.sqrt(delta_nz)
        support_size = torch.sum(tau <= inputs_sorted, dim=axis, keepdim=True)
        tau_star = tau.gather(axis, support_size - 1)
        return tau_star, support_size

    def set_params(self, **kwargs):
        self.config = {}
        self.config.update(kwargs)

        if self.config['n_estimators'] == 1:
            self.config['selected_variables'] = 1.0
            self.config['data_subset_fraction'] = 1.0
            self.config['dropout'] = 0.0
            self.config['bootstrap'] = False
        
        for arg_key, arg_value in self.config.items():
            setattr(self, arg_key, arg_value)

    def preprocess_data(self, X_train, y_train, X_val, y_val):
        if isinstance(y_train, pd.Series):
            y_train = y_train.values.astype(np.float64)
        if isinstance(y_val, pd.Series):
            y_val = y_val.values.astype(np.float64)

        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train)
        if not isinstance(X_val, pd.DataFrame):
            X_val = pd.DataFrame(X_val)

        binary_indices = []
        low_cardinality_indices = []
        high_cardinality_indices = []
        num_columns = []
        for column_index, column in enumerate(X_train.columns):
            if column_index in self.cat_idx:
                if len(X_train.iloc[:, column_index].unique()) <= 2:
                    binary_indices.append(column)
                if len(X_train.iloc[:, column_index].unique()) < 5:
                    low_cardinality_indices.append(column)
                else:
                    high_cardinality_indices.append(column)
            else:
                num_columns.append(column)

        cat_columns = [col for col in X_train.columns if col not in num_columns]

        if len(num_columns) > 0:
            self.mean_train_num = X_train[num_columns].mean(axis=0)
            X_train[num_columns] = X_train[num_columns].fillna(self.mean_train_num)
            X_val[num_columns] = X_val[num_columns].fillna(self.mean_train_num)
        if len(cat_columns) > 0:
            self.mode_train_cat = X_train[cat_columns].mode(axis=0).iloc[0]
            X_train[cat_columns] = X_train[cat_columns].fillna(self.mode_train_cat)
            X_val[cat_columns] = X_val[cat_columns].fillna(self.mode_train_cat)

        self.cat_columns = cat_columns
        self.num_columns = num_columns
        
        if(len(self.cat_columns) != 0):
            self.encoder_ordinal = ce.OrdinalEncoder(cols=binary_indices)
            self.encoder_ordinal.fit(X_train)
            X_train = self.encoder_ordinal.transform(X_train)
            X_val = self.encoder_ordinal.transform(X_val)

            self.encoder_loo = ce.LeaveOneOutEncoder(cols=high_cardinality_indices)
            self.encoder_loo.fit(X_train, y_train)
            X_train = self.encoder_loo.transform(X_train)
            X_val = self.encoder_loo.transform(X_val)

            self.encoder_ohe = ce.OneHotEncoder(cols=low_cardinality_indices)
            self.encoder_ohe.fit(X_train)
            X_train = self.encoder_ohe.transform(X_train)
            X_val = self.encoder_ohe.transform(X_val)

        X_train = X_train.astype(np.float64)
        X_val = X_val.astype(np.float64)
        X_train = X_train.values
        X_val = X_val.values
        # quantile_noise = 1e-4
        # quantile_train = np.copy(X_train.values).astype(np.float64)
        
        # stds = np.std(quantile_train, axis=0, keepdims=True)
        # noise_std = quantile_noise / np.maximum(stds, quantile_noise)
        # quantile_train += noise_std * np.random.randn(*quantile_train.shape)

        # quantile_train = pd.DataFrame(quantile_train, columns=X_train.columns, index=X_train.index)

        # self.normalizer = sklearn.preprocessing.QuantileTransformer(
        #     n_quantiles=min(quantile_train.shape[0], 1000),
        #     output_distribution='normal',
        # )

        # self.normalizer.fit(quantile_train.values.astype(np.float64))
        # X_train = self.normalizer.transform(X_train.values.astype(np.float64))
        # X_val = self.normalizer.transform(X_val.values.astype(np.float64))

        # self.mean = np.mean(y_train)
        # self.std = np.std(y_train)
        # print(self.mean, self.std)

        return X_train, y_train, X_val, y_val

    def apply_preprocessing(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if len(self.num_columns) > 0:
            X[self.num_columns] = X[self.num_columns].fillna(self.mean_train_num)
        if len(self.cat_columns) > 0:
            X[self.cat_columns] = X[self.cat_columns].fillna(self.mode_train_cat)

        X = self.encoder_ordinal.transform(X)
        X = self.encoder_loo.transform(X)
        X = self.encoder_ohe.transform(X)

        X = self.normalizer.transform(X.values.astype(np.float64))

        return X