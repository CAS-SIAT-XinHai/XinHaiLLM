const {defineConfig} = require('@vue/cli-service')
module.exports = defineConfig({
    transpileDependencies: true,
    chainWebpack: config => {
        config
            .plugin('html')
            .tap(args => {
                args[0].title = 'CAS-SIAT-Fintech: AutoInvoice'
                return args
            })
    },
    outputDir: 'dist',
    assetsDir: 'static',
    // baseUrl: IS_PRODUCTION
    // ? 'http://cdn123.com'
    // : '/',
    // For Production, replace set baseUrl to CDN
    // And set the CDN origin to `yourdomain.com/static`
    // Whitenoise will serve once to CDN which will then cache
    // and distribute
    devServer: {
        proxy: {
            '/api*': {
                // Forward frontend dev server request for /api to flask dev server
                target: 'http://localhost:5000/',
                ws: true
            }
        }
    }
})
