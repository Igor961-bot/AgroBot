/** @type {import('tailwindcss').Config} */
module.exports = {
    content: ["./src/**/*.{js,jsx,ts,tsx}"],
    theme: {
        extend: {
            colors: {
                brand: {
                    DEFAULT: '#25ab4d',
                    light: '#EEFEE8',
                    dark: '#5DC17B',
                },
            },
        },
    },
    plugins: [],
};
