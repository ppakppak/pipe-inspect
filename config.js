// API 설정
const API_CONFIG = {
    // 로컬 모드: GPU 서버와 같은 머신
    // 원격 모드: 클라이언트 PC에서 프록시 사용
    BASE_URL: process.env.API_URL || 'http://localhost:5001'
};

// Node.js 환경에서 사용
if (typeof module !== 'undefined' && module.exports) {
    module.exports = API_CONFIG;
}
