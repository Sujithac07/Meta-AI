const fs = require('fs');
const path = require('path');

const dir = r'C:\Users\sujit\OneDrive\Desktop\meta_ai_builder_pp\frontend\react-dashboard';
const indexFile = path.join(dir, 'index.html');
const newFile = path.join(dir, 'dashboard_new.html');

// Delete old index.html
if (fs.existsSync(indexFile)) {
    fs.unlinkSync(indexFile);
    console.log('✓ Deleted old index.html');
} else {
    console.log('⚠ Old index.html not found');
}

// Copy dashboard_new.html to index.html
if (fs.existsSync(newFile)) {
    fs.copyFileSync(newFile, indexFile);
    console.log('✓ Renamed dashboard_new.html to index.html');
} else {
    console.log('✗ dashboard_new.html not found');
}

// Verify new index.html
if (fs.existsSync(indexFile)) {
    const stats = fs.statSync(indexFile);
    console.log(`✓ New index.html exists (${stats.size} bytes)`);
    
    const content = fs.readFileSync(indexFile, 'utf-8');
    const lines = content.split('\n').slice(0, 5);
    console.log('\nFirst 5 lines of new index.html:');
    lines.forEach((line, i) => {
        console.log(`  ${i + 1}. ${line}`);
    });
} else {
    console.log('✗ index.html verification failed');
}
