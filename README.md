# Project Melampo

Melampo reads the documents of a clinical case (notes, reports, laboratory tables, images) and proposes a
structured differential diagnosis with its evidence, sending to a person every reading it cannot confirm.

> **Research software, not a medical device.** It must not be used to diagnose or treat patients.

**Documentation:** [English](docs/en/README.md) · [Italiano](docs/it/README.md)

| | English | Italiano |
|---|---|---|
| What it is, where it stands | [README](docs/en/README.md) | [README](docs/it/README.md) |
| Architecture and data | [ARCHITECTURE](docs/en/ARCHITECTURE.md) | [ARCHITECTURE](docs/it/ARCHITECTURE.md) |
| Validation and data lineage | [VALIDATION](docs/en/VALIDATION.md) | [VALIDATION](docs/it/VALIDATION.md) |
| Compliance and risk | [COMPLIANCE_RISK](docs/en/COMPLIANCE_RISK.md) | [COMPLIANCE_RISK](docs/it/COMPLIANCE_RISK.md) |
| Contributing | [CONTRIBUTING](docs/en/CONTRIBUTING.md) | [CONTRIBUTING](docs/it/CONTRIBUTING.md) |
| Radiologists' guide | [RADIOLOGIST_GUIDE](docs/en/RADIOLOGIST_GUIDE.md) | [RADIOLOGIST_GUIDE](docs/it/RADIOLOGIST_GUIDE.md) |
| Security policy | [SECURITY](docs/SECURITY.md) | |

```bash
uv sync --extra dev --extra data && uv run dvc update -R data && uv run pytest -q
```

## ⚖️ License and Terms of Use

This software is licensed under the **GNU Affero General Public License v3** (GNU AGPLv3). 

To view a copy of this license, visit: https://gnu.org

### SPECIFIC ATTRIBUTION REQUIREMENTS (Pursuant to Section 7(b) of the GNU AGPLv3)

As the Licensor of this software, the Author exercises the right to supplement the terms of this License with additional requirements regarding author attribution. Any user, redistributor, or developer of derivative works or network-accessible applications MUST strictly comply with the following enforcement conditions:

1. **Graphical User Interface (GUI) and Dashboard Attribution:** If the software, or any work derived from it (including web applications and network services), features a Graphical User Interface (GUI) or a web dashboard, a clear, permanent, and easily accessible credit to the original Author and source repository MUST be displayed within the user interface itself. This notice must be visible to the end-user without requiring interaction (e.g., prominently placed in the application footer, main dashboard layout, or the primary "About/Credits" screen).
2. **Required Text Format:** The attribution within the GUI must include the following clickable hyperlink text (or a substantially identical equivalent):
   *"Based on MelampoAI by Francesco Lattari — Source: https://github.com/francescoluciolattari/Melampo-AI"*
3. **Source Code Preservation:** All existing copyright notices, author credits, and license files within the source code must be preserved intact and unaltered in all copies, modifications, or network deployments.

**Failure to comply with these specific attribution terms constitutes a material breach of the license agreement, results in the immediate termination of your rights under this license, and represents an infringement of copyright.**


**THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.

See `LICENSE`.

## Contact

**Francesco Lattari** - Project Lead & Author  
Email: [flattari@chandra.it](mailto:flattari@chandra.it)  
Website: [www.chandra.it](https://www.chandra.it)

---

© 2026 Project Melampo Research Team
