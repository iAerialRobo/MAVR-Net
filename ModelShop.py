class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.spat_feature = spat_model  # feature_size =   Nx512x7x7
        self.temp_feature = temp_model  # feature_size = Nx512x7x7
        self.layer1 = nn.Sequential(nn.Conv3d(1024, 512, 1, stride=1, padding=1, dilation=1, bias=True),
                                    nn.ReLU(), nn.MaxPool3d(kernel_size=2, stride=2))
        self.fc = nn.Sequential(nn.Linear(8192, 2048), nn.ReLU(), nn.Dropout(p=0.85),
                                nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(p=0.85),
                                nn.Linear(512, 101))

    def forward(self, spat_data, temp_data):
        x1 = self.spat_feature(spat_data)
        x2 = self.temp_feature(temp_data)

        y = torch.cat((x1, x2), dim=1)
        for i in range(x1.size(1)):
            y[:, (2 * i), :, :] = x1[:, i, :, :]
            y[:, (2 * i + 1), :, :] = x2[:, i, :, :]

        y = y.view(y.size(0), 1024, 1, 7, 7)
        cnn_out = self.layer1(y)
        cnn_out = cnn_out.view(cnn_out.size(0), -1)
        out = self.fc(cnn_out)
        return out



late fusion

  # Run the forward pass
    outputs_spatial = model_spatial(local_batch_spatial)
    outputs_temporal = model_temporal(local_batch_flow)
    outputs_sum = outputs_spatial + outputs_temporal
    loss_spatial = criterion_spatial(outputs_spatial, local_labels)
    loss_temporal = criterion_temporal(outputs_temporal, local_labels)
    loss_total = loss_spatial + loss_temporal
    loss_list_spatial.append(loss_spatial.item())
    loss_list_temporal.append(loss_temporal.item())

    # Backprop and perform Adam optimisation
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()

    # Track the accuracy
    total = local_labels.size(0)
    _, predicted = torch.max(outputs_sum.data, 1)
    correct = (predicted == local_labels).sum().item()
    predicted = predicted.cpu().detach().numpy()
    local_labels = local_labels.cpu().detach().numpy()
    print(predicted)
    for item in predicted:
        predicted_values.append(item)
    for item in local_labels:
        actual.append(item)
    print(local_labels)
    acc_list.append(correct / total)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1_spatial = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer1_flow = nn.Sequential(
            nn.Conv2d(18, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(401408, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x, y):
        x1 = self.layer1_spatial(x)
        x1 = x1.view(x1.size(0), -1)
        x2 = self.layer1_flow(y)
        x2 = x2.view(x2.size(0), -1)
        fusion = torch.cat((x1, x2), dim=1)
        out = fusion.reshape(fusion.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out