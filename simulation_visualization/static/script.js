document.addEventListener('DOMContentLoaded', function () {
    const configSelect = document.getElementById('config-select');
    const startSimulationButton = document.getElementById('start-simulation');
    const chatHistory = document.getElementById('chat-history');
    const agentDetails = document.getElementById('agent-details');
    let latestIndexIds = {};
    let agentIds = [];
    let agentAvatars = {};

    // 获取配置文件列表
    fetch('/get_configs')
        .then(response => response.json())
        .then(data => {
            const defaultOption = document.createElement('option');
            defaultOption.value = "";
            defaultOption.textContent = "请选择配置文件";
            defaultOption.disabled = true;
            defaultOption.selected = true;
            configSelect.appendChild(defaultOption);

            data.forEach(file => {
                const option = document.createElement('option');
                option.value = file;
                option.textContent = file;
                configSelect.appendChild(option);
            });
        });

    // 监听选择配置文件的变化
    configSelect.addEventListener('change', function () {
        const selectedConfig = this.value;
        if (selectedConfig) {
            fetch(`/read_config?filename=${selectedConfig}`)
                .then(response => response.json())
                .then(data => {
                    updateAgentsInfo(data);
                    extractAgentIdsFromEdges(data.arena.environment.topology.edges);
                    getLatestIndexIds(agentIds.length);
                });
        }
    });

    // 启动模拟
    startSimulationButton.addEventListener('click', function () {
        const selectedConfig = configSelect.value;
        if (selectedConfig) {
            fetch('/start_simulation', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ config_file: selectedConfig })
            })
                .then(response => response.json())
                .then(data => {
                    console.log('Simulation started:', data);
                    alert("模拟已开始!");
                    startFetchingHistory();
                })
                .catch(error => {
                    console.error('Error starting simulation:', error);
                    alert("启动模拟失败，请检查日志!");
                });
        } else {
            alert("请选择一个配置文件来启动模拟。");
        }
    });

    // 停止模拟
    window.addEventListener('beforeunload', function (e) {
        fetch('/stop_simulation', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            console.log('Simulation stopped:', data);
        })
        .catch(error => {
            console.error('Error stopping simulation:', error);
        });
    });

    // 提取智能体ID列表
    function extractAgentIdsFromEdges(edges) {
        const agentSet = new Set();
        edges.forEach(edge => {
            const [from, to] = edge.split('->').map(Number);
            agentSet.add(from);
            agentSet.add(to);
        });
        agentIds = Array.from(agentSet);
    }

    // 获取最新的indexId
    function getLatestIndexIds(agentCount) {
        latestIndexIds = {};
        agentIds.forEach(agentId => {
            fetch(`/fetch_memory/${agentId}`)
                .then(response => response.json())
                .then(data => {
                    const messages = data.memory.short_term_memory.messages;
                    if (messages.length > 0) {
                        latestIndexIds[agentId] = messages[messages.length - 1].indexId;
                    } else {
                        latestIndexIds[agentId] = "0";
                    }
                });
        });
    }

    // 定时刷新对话历史
    function startFetchingHistory() {
        setInterval(() => {
            agentIds.forEach(agentId => {
                fetch(`/fetch_memory/${agentId}`)
                    .then(response => response.json())
                    .then(data => {
                        const messages = data.memory.short_term_memory.messages;
                        messages.forEach(message => {
                            if (parseInt(message.indexId) > parseInt(latestIndexIds[agentId])) {
                                const newMessage = document.createElement('div');
                                newMessage.className = 'chat-message';
                                newMessage.dataset.indexId = message.indexId;

                                const avatarUrl = agentAvatars[agentId];
                                newMessage.innerHTML = `
                                    <img src="${avatarUrl}" alt="${message.username}">
                                    <div class="message-content">
                                        <h5>${message.username} (${message.timestamp}):</h5>
                                        <p>${message.content}</p>
                                    </div>
                                `;
                                chatHistory.appendChild(newMessage);
                                chatHistory.scrollTop = chatHistory.scrollHeight;
                                latestIndexIds[agentId] = message.indexId;
                            }
                        });
                    });
            });
        }, 1000); // 刷新间隔设置为1秒
    }

    // 更新代理信息的显示
    function updateAgentsInfo(configData) {
        agentDetails.innerHTML = '';
        agentAvatars = {};
        configData.arena.agents.forEach(agent => {
            const agentDiv = document.createElement('div');
            agentDiv.className = 'agent';
            const avatarUrl = `https://api.multiavatar.com/${agent.name}.png`;
            agentAvatars[agent.agent_id] = avatarUrl;

            agentDiv.innerHTML = `
                <img src="${avatarUrl}" alt="${agent.name}">
                <div class="name">${agent.name}</div>
            `;

            agentDetails.appendChild(agentDiv);
        });

        // 更新选中的配置选项，存储代理数量
        configSelect.selectedOptions[0].setAttribute('data-agents-count', configData.arena.agents.length);
    }
});
