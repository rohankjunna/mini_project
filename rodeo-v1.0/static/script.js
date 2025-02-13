// Create a glowing effect element
const glowingEffect = document.createElement('div');
glowingEffect.classList.add('glowing-effect');
document.body.appendChild(glowingEffect);

// Add event listeners to track the cursor position
document.addEventListener('mousemove', (event) => {
    glowingEffect.style.setProperty('--x', `${event.clientX}px`);
    glowingEffect.style.setProperty('--y', `${event.clientY}px`);
});

// Add a class to the glowing effect element to enable JavaScript-generated styles
glowingEffect.classList.add('js-generated');

const imageInput = document.getElementById('image-input');
const generateCaptionBtn = document.getElementById('generate-caption-btn');
const removeBackgroundBtn = document.getElementById('remove-background-btn');
const changeBackgroundBtn = document.getElementById('change-background-btn');
const outputDiv = document.getElementById('output');

generateCaptionBtn.addEventListener('click', async () => {
    try {
        const imgPath = imageInput.files[0].path;
        const response = await fetch('/generate_caption', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ img_path: imgPath }),
        });
        const caption = await response.json();
        outputDiv.innerHTML = `Generated Caption: ${caption.caption}`;
    } catch (error) {
        console.error(error);
        outputDiv.innerHTML = 'Error generating caption';
    }
});

removeBackgroundBtn.addEventListener('click', async () => {
    try {
        const imgPath = imageInput.files[0].path;
        const response = await fetch('/remove_background', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ img_path: imgPath }),
        });
        const outputPath = await response.json();
        outputDiv.innerHTML = `Removed Background: ${outputPath.output_path}`;
    } catch (error) {
        console.error(error);
        outputDiv.innerHTML = 'Error removing background';
    }
});

changeBackgroundBtn.addEventListener('click', async () => {
    try {
        const foregroundPath = imageInput.files[0].path;
        const backgroundPath = 'path/to/background/image.jpg'; // Replace with the actual background image path
        const response = await fetch('/change_background', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ foreground_path: foregroundPath, background_path: backgroundPath }),
        });
        const outputPath = await response.json();
        outputDiv.innerHTML = `Changed Background: ${outputPath.output_path}`;
    } catch (error) {
        console.error(error);
        outputDiv.innerHTML = 'Error changing background';
    }
});