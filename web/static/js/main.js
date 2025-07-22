/**
 * 联邦学习结果可视化系统 - 主要JavaScript功能
 * Main JavaScript functionality for Federated Learning Results Visualization System
 */

// 全局变量
let charts = {};
let currentData = null;

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

/**
 * 初始化应用
 */
function initializeApp() {
    // 初始化文件上传功能
    initializeFileUpload();
    
    // 初始化工具提示
    initializeTooltips();
    
    // 初始化响应式表格
    initializeResponsiveTables();
}

/**
 * 初始化文件上传功能
 */
function initializeFileUpload() {
    const uploadArea = document.querySelector('.upload-area');
    const fileInput = document.querySelector('input[type="file"]');
    
    if (!uploadArea || !fileInput) return;
    
    // 拖拽上传功能
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadArea.classList.add('drag-over');
    });
    
    uploadArea.addEventListener('dragleave', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
    });
    
    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
        
        const files = e.dataTransfer.files;
        if (files.length > 0 && files[0].name.endsWith('.json')) {
            fileInput.files = files;
            // 自动提交表单
            const form = fileInput.closest('form');
            if (form) {
                form.submit();
            }
        } else {
            showNotification('请选择有效的JSON文件', 'warning');
        }
    });
    
    // 文件选择验证
    fileInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file && !file.name.endsWith('.json')) {
            showNotification('请选择JSON格式的文件', 'warning');
            e.target.value = '';
        }
    });
}

/**
 * 初始化工具提示
 */
function initializeTooltips() {
    // 如果使用Bootstrap，初始化工具提示
    if (typeof bootstrap !== 'undefined') {
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }
}

/**
 * 初始化响应式表格
 */
function initializeResponsiveTables() {
    const tables = document.querySelectorAll('.table-responsive table');
    tables.forEach(table => {
        // 为小屏幕添加滚动提示
        if (window.innerWidth < 768) {
            const wrapper = table.closest('.table-responsive');
            if (wrapper) {
                wrapper.setAttribute('title', '左右滑动查看更多内容');
            }
        }
    });
}

/**
 * 显示通知消息
 */
function showNotification(message, type = 'info', duration = 3000) {
    const notification = document.createElement('div');
    notification.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
    notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
    notification.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(notification);
    
    // 自动移除通知
    setTimeout(() => {
        if (notification.parentNode) {
            notification.remove();
        }
    }, duration);
}

/**
 * 格式化数字显示
 */
function formatNumber(num, decimals = 4) {
    if (typeof num !== 'number') return num;
    return num.toFixed(decimals);
}

/**
 * 格式化百分比显示
 */
function formatPercentage(num, decimals = 2) {
    if (typeof num !== 'number') return num;
    return (num * 100).toFixed(decimals) + '%';
}

/**
 * 获取随机颜色
 */
function getRandomColor(index) {
    const colors = [
        'rgb(255, 99, 132)', 'rgb(54, 162, 235)', 'rgb(255, 205, 86)',
        'rgb(75, 192, 192)', 'rgb(153, 102, 255)', 'rgb(255, 159, 64)',
        'rgb(199, 199, 199)', 'rgb(83, 102, 147)', 'rgb(255, 99, 255)',
        'rgb(99, 255, 132)', 'rgb(255, 132, 99)', 'rgb(132, 99, 255)'
    ];
    return colors[index % colors.length];
}

/**
 * 创建图表配置
 */
function createChartConfig(type, data, options = {}) {
    const defaultOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                position: 'top',
            },
            tooltip: {
                mode: 'index',
                intersect: false,
                backgroundColor: 'rgba(0, 0, 0, 0.8)',
                titleColor: 'white',
                bodyColor: 'white',
                borderColor: 'rgba(255, 255, 255, 0.1)',
                borderWidth: 1
            }
        },
        scales: {
            x: {
                display: true,
                title: {
                    display: true,
                    text: '训练轮次'
                }
            },
            y: {
                display: true,
                beginAtZero: true
            }
        },
        interaction: {
            mode: 'nearest',
            axis: 'x',
            intersect: false
        }
    };
    
    return {
        type: type,
        data: data,
        options: Object.assign(defaultOptions, options)
    };
}

/**
 * 销毁所有图表
 */
function destroyAllCharts() {
    Object.values(charts).forEach(chart => {
        if (chart && typeof chart.destroy === 'function') {
            chart.destroy();
        }
    });
    charts = {};
}

/**
 * 导出图表为图片
 */
function exportChart(chartId, filename) {
    const chart = charts[chartId];
    if (!chart) {
        showNotification('图表不存在', 'error');
        return;
    }
    
    const canvas = chart.canvas;
    const url = canvas.toDataURL('image/png');
    
    const link = document.createElement('a');
    link.download = filename || 'chart.png';
    link.href = url;
    link.click();
}

/**
 * 切换图表类型
 */
function toggleChartType(chartId, newType) {
    const chart = charts[chartId];
    if (!chart) return;
    
    chart.config.type = newType;
    chart.update();
}

/**
 * 处理窗口大小变化
 */
window.addEventListener('resize', function() {
    // 重新调整图表大小
    Object.values(charts).forEach(chart => {
        if (chart && typeof chart.resize === 'function') {
            chart.resize();
        }
    });
});

/**
 * 页面卸载时清理资源
 */
window.addEventListener('beforeunload', function() {
    destroyAllCharts();
});

// 导出全局函数供其他脚本使用
window.FedLearningViz = {
    showNotification,
    formatNumber,
    formatPercentage,
    getRandomColor,
    createChartConfig,
    exportChart,
    toggleChartType,
    charts
};